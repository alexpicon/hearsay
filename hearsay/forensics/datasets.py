# Author: Alex Picon <alexnpc@me.com>
"""Labelled corpora used to measure and calibrate the forensic experts.

* ``lj_diffssd``: the organisers' training data, LJ Speech (real, one female
  speaker) against a fixed sample of DiffSSD generators (synthetic).
* ``la19``: a balanced subset of the ASVspoof 2019 LA evaluation partition
  (VCTK bona fide speakers against attacks A07-A19).
* ``*_channel``: the same clips passed through a simulated copy of the test
  channel (colored noise and rumble, low-pass near 7.4 kHz, peak
  normalisation), so we can see which evidence survives it.
* ``*_trim_channel``: as above, but first trimmed like the test pipeline
  (librosa-style trim at 22.05 kHz, 512-sample hop). This matters: ASVspoof
  2019 bona fide clips start with about 1 s of silence and spoofs with about
  0.07 s, a known shortcut that the test set's trimming removes.

Labels are 1 for synthetic and 0 for real. Audio stays outside the repo.
"""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)
DATA = Path(__file__).resolve().parents[2] / "_internal" / "hearsay"
SIM = DATA / "sim"
GENERATORS_PER_SET = 50

Item = tuple[Path, int, str]


def lj_diffssd(per_generator: int = GENERATORS_PER_SET) -> list[Item]:
    """List LJ Speech (real) and a deterministic DiffSSD sample (synthetic).

    Args:
        per_generator: Clips to take from each DiffSSD generator.

    Returns:
        (path, label, group) items; group is ``lj`` or the generator name.

    """
    items: list[Item] = [(p, 0, "lj") for p in sorted((DATA / "lj").rglob("*.wav"))]
    root = DATA / "diffssd" / "DiffSSD" / "generated_speech"
    rng = random.Random(13)
    for gen in sorted(d for d in root.iterdir() if d.is_dir()):
        files = sorted(p for p in gen.rglob("*") if p.suffix in (".wav", ".mp3"))
        items += [
            (p, 1, gen.name) for p in rng.sample(files, min(per_generator, len(files)))
        ]
    return items


def la19() -> list[Item]:
    """List the extracted ASVspoof 2019 LA evaluation subset.

    Returns:
        (path, label, group) items; group is ``bonafide`` or the attack id.

    """
    meta_path = DATA / "asvspoof2019" / "wav" / "meta.json"
    meta = json.loads(meta_path.read_text())
    root = meta_path.parent
    return [
        (root / m["label"] / f"{m['uid']}.wav", int(m["label"] == "spoof"),
         "bonafide" if m["label"] == "bonafide" else m["attack"])
        for m in meta
    ]  # fmt: skip


def _channel_one(args: tuple[str, str, int]) -> str:
    """Pass one file through the simulated test channel.

    Args:
        args: (source path, destination path, seed, trim first).

    Returns:
        Destination path.

    """
    from forensics.audio import load_audio
    from hearsay.augment import ChannelConfig, simulate_channel

    src, dst, seed, trim = args
    y, _ = load_audio(Path(src), 16000)
    if trim:
        import librosa

        wide = librosa.resample(y, orig_sr=16000, target_sr=22050)
        wide, _ = librosa.effects.trim(
            wide, top_db=40, frame_length=2048, hop_length=512
        )
        y = librosa.resample(wide, orig_sr=22050, target_sr=16000).astype(np.float32)
    cfg = ChannelConfig(
        p_noise=1.0, snr_db=(12.0, 32.0), p_lowpass=1.0, cutoff_hz=(7200.0, 7600.0),
        p_codec=0.0, p_resample=0.0,
    )  # fmt: skip
    out = simulate_channel(y, np.random.default_rng(seed), cfg)
    sf.write(dst, out, 16000, subtype="PCM_16")
    return dst


def channel(
    items: list[Item],
    name: str,
    trim: bool = False,
    jobs: int = 16,
) -> list[Item]:
    """Create (once) channel-degraded copies of a labelled set.

    Args:
        items: Source items.
        name: Output folder name under the simulation directory.
        trim: Trim leading/trailing silence first (test-like).
        jobs: Worker processes.

    Returns:
        Items pointing at the degraded copies (same labels and groups). Clips
        whose generator and file name collide (the same sentence from two
        speakers) are kept once.

    """
    out_dir = SIM / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Item] = []
    todo = []
    seen: set[Path] = set()
    for i, (path, label, group) in enumerate(items):
        dst = out_dir / f"{group}__{path.stem}.wav"
        if dst in seen:
            continue
        seen.add(dst)
        out.append((dst, label, group))
        if not dst.exists():
            todo.append((str(path), str(dst), i, trim))
    if todo:
        log.info("simulating channel for %d files into %s", len(todo), out_dir)
        with ProcessPoolExecutor(jobs) as ex:
            list(ex.map(_channel_one, todo, chunksize=8))
    return out


def load(name: str) -> list[Item]:
    """Return a labelled set by name.

    Args:
        name: ``lj_diffssd`` or ``la19``, optionally with a ``_channel`` or
            ``_trim_channel`` suffix.

    Returns:
        The items.

    """
    base = {"lj_diffssd": lj_diffssd, "la19": la19}
    if name in base:
        return base[name]()
    for suffix, trim in (("_trim_channel", True), ("_channel", False)):
        stem = name.removesuffix(suffix)
        if name.endswith(suffix) and stem in base:
            return channel(base[stem](), name, trim)
    msg = f"unknown dataset {name!r}"
    raise KeyError(msg)
