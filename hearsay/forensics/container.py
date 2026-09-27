# Author: Alex Picon <alexnpc@me.com>
"""Container, metadata and file-system forensics (ffprobe, exiftool, RIFF).

This expert never decodes audio. It reads what the file *claims* about itself
(codec chain, encoder tags, device and date tags, header fields, MAC times)
and checks that the claims are consistent. Its score is rule based on purpose:
on labelled corpora a learned container model would only learn which dataset
a file came from, not whether the voice is real.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from forensics.calibration import sigmoid
from forensics.finding import Context, Finding
from forensics.riff import length_grids, parse_riff

SYNTH_TOOLS = (
    "elevenlabs", "play.ht", "playht", "coqui", "xtts", "tortoise", "bark",
    "openvoice", "tacotron", "fastspeech", "vits", "espnet", "resemble",
    "murf", "wellsaid", "polly", "wavenet", "text-to-speech", "tts", "voicebox",
    "speechify", "descript", "rvc", "so-vits",
)  # fmt: skip
DEVICE_KEYS = ("Make", "Model", "Device", "AndroidModel", "Encoder")
# Tags exiftool derives from the file system, not from the file's contents;
# file names must never count as evidence.
FILESYSTEM_TAGS = frozenset(
    {
        "SourceFile", "FileName", "Directory", "FileSize", "FileModifyDate",
        "FileAccessDate", "FileInodeChangeDate", "FilePermissions", "FileType",
        "FileTypeExtension", "MIMEType", "ExifToolVersion",
    }
)  # fmt: skip
LAVF_RELEASE = {"58.29": "2019-08-05", "58.76": "2021-04-17", "59.27": "2022-07-22"}
LAVF_RELEASE |= {"60.16": "2023-11-10", "61.7": "2024-09-29", "62.3": "2025-08-22"}
DESCRIPTIONS = {
    "synth_tool_tag": "metadata names a speech-synthesis tool",
    "header_issues": "RIFF header inconsistencies",
    "vocoder_grid": "sample count sits on a vocoder frame grid",
    "timestamp_issues": "impossible or contradictory timestamps",
    "device_claim": "metadata claims a recording device",
    "lossy_codec": "lossy codec in the container",
}


def _run_json(cmd: list[str]) -> Any:
    """Run a command that prints JSON and parse its output.

    Args:
        cmd: Command line.

    Returns:
        Parsed JSON, or an empty dict when the tool fails.

    """
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout or "{}")
    except subprocess.SubprocessError, json.JSONDecodeError, OSError:
        return {}


def probe(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ffprobe and exiftool metadata for a file.

    Args:
        path: Audio file.

    Returns:
        (ffprobe JSON, exiftool tag dictionary).

    """
    ff = _run_json(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", str(path)]
    )  # fmt: skip
    ex = _run_json(["exiftool", "-j", "-api", "largefilesupport=1", str(path)])
    return ff, (ex[0] if isinstance(ex, list) and ex else {})


def _mac_times(path: Path) -> dict[str, str]:
    """Return modified/accessed/changed times as ISO strings.

    Args:
        path: File on disk.

    Returns:
        MAC timestamps (Linux reports inode change, not creation, as ``ctime``).

    """
    st = path.stat()
    iso = {
        "modified": st.st_mtime,
        "accessed": st.st_atime,
        "changed": st.st_ctime,
    }
    return {k: dt.datetime.fromtimestamp(v).isoformat() for k, v in iso.items()}


def _timestamp_issues(path: Path, encoder: str, tags: dict[str, Any]) -> list[str]:
    """List timestamp contradictions (future dates, encoder newer than file).

    Args:
        path: File on disk.
        encoder: Encoder tag (for example ``Lavf58.29.100``).
        tags: Exiftool tags.

    Returns:
        Human-readable issues.

    """
    issues = []
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    if mtime > dt.datetime.now() + dt.timedelta(days=1):
        issues.append(f"modification time {mtime:%Y-%m-%d} is in the future")
    if encoder.startswith("Lavf"):
        ver = ".".join(encoder[4:].split(".")[:2])
        released = LAVF_RELEASE.get(ver)
        if released and mtime < dt.datetime.fromisoformat(released):
            issues.append(f"file dated {mtime:%Y-%m-%d} before {encoder} existed")
    for key in ("CreateDate", "DateTimeOriginal", "DateCreated"):
        value = str(tags.get(key, ""))
        if value[:4].isdigit() and int(value[:4]) > mtime.year + 1:
            issues.append(f"{key} {value} is later than the file's mtime")
    return issues


def _sample_rate(stream: dict[str, Any], riff: dict[str, Any]) -> int:
    """Prefer valid probe metadata, falling back to the native WAV header."""
    for value in (stream.get("sample_rate"), riff.get("fmt", {}).get("sr")):
        try:
            rate = int(value)
        except TypeError, ValueError, OverflowError:
            continue
        if rate > 0:
            return rate
    return 0


def analyze(path: Path, ctx: Context) -> Finding:
    """Inspect the container and metadata of one file.

    Args:
        path: Audio file.
        ctx: File context (``batch`` may mark grids shared by the whole set).

    Returns:
        The container finding.

    """
    started = time.perf_counter()
    ff, tags = probe(path)
    streams = [s for s in ff.get("streams", []) if s.get("codec_type") == "audio"]
    stream = streams[0] if streams else {}
    fmt = ff.get("format", {})
    encoder = str(fmt.get("tags", {}).get("encoder", tags.get("Software", "")))
    riff = parse_riff(path) if path.suffix.lower() == ".wav" else {}
    native = riff.get("fmt", {})
    codec = str(
        stream.get("codec_name")
        or {1: "pcm", 3: "pcm_float"}.get(native.get("tag"), "unknown")
    )
    sr = _sample_rate(stream, riff)
    channels = stream.get("channels") or native.get("channels", "?")
    n = int(riff.get("n_samples", 0) or 0)
    grids = length_grids(n, sr) if n else []
    shared = set(ctx.batch.get("shared_grids", []))
    own_grids = [g for g in grids if g not in shared]
    if shared & set(grids):
        own_grids = [g for g in own_grids if not g.endswith("~")]
    embedded = {k: v for k, v in tags.items() if k not in FILESYSTEM_TAGS}
    text = " ".join(str(v) for v in embedded.values()).lower()
    synth = sorted({t for t in SYNTH_TOOLS if f" {t}" in f" {text}"} - {"tts"})
    device = {k: tags[k] for k in DEVICE_KEYS if k in tags and k != "Encoder"}
    ts_issues = _timestamp_issues(path, encoder, tags)
    feats = {
        "synth_tool_tag": float(bool(synth)),
        "header_issues": float(len(riff.get("issues", []))),
        "vocoder_grid": 1.0
        if any(not g.endswith("~") for g in own_grids)
        else 0.4 * bool(own_grids),
        "timestamp_issues": float(len(ts_issues)),
        "device_claim": float(bool(device)),
        "lossy_codec": float(
            codec != "unknown" and not codec.startswith(("pcm", "flac", "alac"))
        ),
        "native_sr": float(sr),
        "duration_s": float(fmt.get("duration", n / sr if sr else "nan")),
    }
    weights = {
        "synth_tool_tag": 4.0,
        "header_issues": 0.8,
        "vocoder_grid": 1.2,
        "timestamp_issues": 1.0,
        "device_claim": -0.4,
    }
    contrib = {k: w * min(2.0, feats[k]) for k, w in weights.items() if feats[k]}
    evidence: dict[str, Any] = {
        "codec_chain": f"{codec} {sr} Hz {channels} ch in "
        f"{fmt.get('format_name', '?')}, encoder '{encoder or 'none'}'",
        "encoder": encoder,
        "bit_rate": fmt.get("bit_rate"),
        "riff_chunks": riff.get("chunks", []),
        "header_issues": riff.get("issues", []),
        "length_grids": grids,
        "grids_shared_by_batch": sorted(set(grids) & shared),
        "synth_tool_tags": synth,
        "device_tags": device,
        "timestamp_issues": ts_issues,
        "mac_times": _mac_times(path),
        "exif_software": tags.get("Software"),
    }
    if contrib:
        score: float | None = sigmoid(sum(contrib.values()))
        reasons = [DESCRIPTIONS[k] for k in contrib]
        summary = "Container evidence: " + ", ".join(reasons) + "."
        confidence = min(0.9, 0.3 + 0.15 * sum(abs(c) for c in contrib.values()))
    else:
        score, confidence = None, 0.0
        summary = f"Clean container ({evidence['codec_chain']}); no metadata red flags."
    if shared & set(grids):
        summary += " Length grid is shared by most of the batch (pipeline, ignored)."
    feats = {k: v for k, v in feats.items() if not math.isnan(v)}
    return Finding(
        expert="container",
        score=score,
        confidence=confidence,
        summary=summary,
        features=feats,
        evidence=evidence,
        contributions=contrib,
        elapsed_ms=1000 * (time.perf_counter() - started),
    )
