# Author: Alex Picon <alexnpc@me.com>
"""Training and test corpora as manifests (one row per clip).

Every manifest has the columns ``uid, source, partition, speaker, system,
label, path, rg, row``. ``label`` is 1 for synthetic and 0 for bona fide.
Clips stored inside parquet files (ASVspoof 2019 LA on Hugging Face) are
addressed by row group and row; clips on disk by ``path``.
"""

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hearsay.audio import AudioDecodeError, load_audio

logger = logging.getLogger(__name__)
DATA_ROOT = Path(
    os.environ.get(
        "HEARSAY_DATA",
        str(Path(__file__).resolve().parents[2] / "_internal" / "hearsay"),
    )
)
LA19_FILES = {
    "train": "ext/la19_train.parquet",
    "dev": "ext/la19_validation.parquet",
    "eval": "ext/la19_test.parquet",
}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm"}
COLUMNS = ["uid", "source", "partition", "speaker", "system", "label", "path"]


def la19_manifest(partition: str) -> pd.DataFrame:
    """Manifest of one ASVspoof 2019 LA partition stored as parquet.

    Args:
        partition: One of ``train``, ``dev`` or ``eval``.

    Returns:
        The manifest (audio is read lazily).

    """
    path = DATA_ROOT / LA19_FILES[partition]
    pf = pq.ParquetFile(path)
    rows = []
    cols = ["speaker_id", "audio_file_name", "system_id", "key"]
    for rg in range(pf.num_row_groups):
        meta = pf.read_row_group(rg, columns=cols).to_pylist()
        for i, r in enumerate(meta):
            rows.append(
                {
                    "uid": r["audio_file_name"],
                    "source": "la19",
                    "partition": partition,
                    "speaker": r["speaker_id"],
                    "system": "bonafide" if r["key"] == 0 else r["system_id"],
                    "label": int(r["key"]),
                    "path": str(path),
                    "rg": rg,
                    "row": i,
                }
            )
    return pd.DataFrame(rows)


def vctk_manifest(pattern: str = "ext/vctk_*.parquet") -> pd.DataFrame:
    """Manifest of downloaded VCTK 0.92 shards (48 kHz, both microphones).

    The test set's real clips have VCTK's own recording noise (a steep
    low-frequency rumble over a very low high-frequency floor) at about the
    same SNR, which the processed ASVspoof 2019 LA bona fide files lack.

    Args:
        pattern: Glob (relative to ``DATA_ROOT``) of the parquet shards.

    Returns:
        The manifest (all bona fide).

    """
    rows = []
    for path in sorted(DATA_ROOT.glob(pattern)):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            meta = pf.read_row_group(rg, columns=["speaker_id", "file"]).to_pylist()
            for i, r in enumerate(meta):
                rows.append(
                    {
                        "uid": Path(r["file"]).stem,
                        "source": "vctk",
                        "partition": "vctk",
                        "speaker": r["speaker_id"],
                        "system": "bonafide",
                        "label": 0,
                        "path": str(path),
                        "rg": rg,
                        "row": i,
                    }
                )
    return pd.DataFrame(rows, columns=[*COLUMNS, "rg", "row"])


def folder_manifest(root: Path, source: str, label: int | None) -> pd.DataFrame:
    """Manifest of every audio file below a folder.

    The generator name is the first folder below ``root`` and the speaker is
    the parent folder when it looks like ``speaker_*``.

    Args:
        root: Folder to scan recursively.
        source: Name of the corpus.
        label: 1 synthetic, 0 bona fide, None unknown (test set).

    Returns:
        The manifest sorted by path.

    """
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXT or p.name.startswith("."):
            continue
        rel = p.relative_to(root).parts
        system = rel[0] if len(rel) > 1 else source
        speaker = p.parent.name if p.parent.name.startswith("speaker_") else system
        rows.append(
            {
                "uid": f"{source}/{'/'.join(rel)}" if label is not None else p.name,
                "source": source,
                "partition": source,
                "speaker": speaker,
                "system": system if label != 0 else "bonafide",
                "label": -1 if label is None else label,
                "path": str(p),
                "rg": -1,
                "row": -1,
            }
        )
    return pd.DataFrame(rows, columns=[*COLUMNS, "rg", "row"])


def diffssd_manifest() -> pd.DataFrame:
    """Manifest of the DiffSSD sample (all synthetic)."""
    return folder_manifest(DATA_ROOT / "diffssd/DiffSSD/generated_speech", "diffssd", 1)


def lj_manifest() -> pd.DataFrame:
    """Manifest of the provided LJ Speech clips (all bona fide)."""
    return folder_manifest(DATA_ROOT / "lj/resampled", "lj", 0)


def test_manifest(folder: Path | None = None) -> pd.DataFrame:
    """Manifest of the unlabeled HEARSAY test clips."""
    return folder_manifest(
        folder or DATA_ROOT / "test/HackGTHearsayTesting", "test", None
    )


def iter_audio(manifest: pd.DataFrame) -> Iterator[tuple[str, np.ndarray | None]]:
    """Yield ``(uid, waveform)`` for every row, reading parquet groups once.

    Args:
        manifest: A manifest; parquet rows should be grouped by ``rg``.

    Yields:
        The uid and the waveform, or None when the clip cannot be decoded.

    """
    cache: tuple[tuple[str, int], list[dict]] | None = None
    for r in manifest.itertuples(index=False):
        try:
            if r.rg >= 0:
                key = (r.path, int(r.rg))
                if cache is None or cache[0] != key:
                    table = pq.ParquetFile(r.path).read_row_group(
                        r.rg, columns=["audio"]
                    )
                    cache = (key, table.column("audio").to_pylist())
                yield r.uid, load_audio(cache[1][int(r.row)]["bytes"])
            else:
                yield r.uid, load_audio(r.path)
        except AudioDecodeError as err:
            logger.warning("cannot decode %s: %s", r.uid, err)
            yield r.uid, None
