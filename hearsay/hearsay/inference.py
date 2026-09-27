# Author: Alex Picon <alexnpc@me.com>
"""Inference API used by the HEARSAY CLI and the optional forensic router."""

import csv
import logging
import os
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import torch

from hearsay.audio import SAMPLE_RATE, AudioDecodeError, load_audio
from hearsay.model import MODEL_PATH, Detector
from hearsay.spectral import spectral_features
from hearsay.ssl import get_embedder
from hearsay.submission import (
    audio_paths,
    validate_filenames,
    validate_probabilities,
)

logger = logging.getLogger(__name__)
MAX_SECONDS = 12.0


class ScoreResult(dict):
    """Result dictionary that also converts to its probability with ``float()``.

    Callers that only need a number (for example the forensic router) can use
    ``float(score_file(path))``; callers that want the evidence read the keys.
    """

    def __float__(self) -> float:
        """Probability that the clip is synthetic."""
        return float(self["probability"])


@cache
def get_detector(path: str | None = None) -> Detector:
    """Load (once per process) the trained detector bundle.

    Args:
        path: Bundle path; defaults to ``$HEARSAY_MODEL`` or the packaged one.

    Returns:
        The detector.

    """
    target = Path(path or os.environ.get("HEARSAY_MODEL", str(MODEL_PATH)))
    logger.info("loading detector %s", target)
    return Detector.load(target)


def features_for(wav: np.ndarray, det: Detector) -> tuple[np.ndarray, np.ndarray]:
    """SSL statistics and hand-crafted features of one waveform.

    Args:
        wav: Waveform at 16 kHz.
        det: Detector (decides the front-end).

    Returns:
        ``(ssl, spec)`` with a leading batch axis of 1.

    """
    emb = get_embedder(det.ssl_model, det.n_layers)
    ssl = emb.embed_one(wav, MAX_SECONDS)[None]
    spec = spectral_features(wav)[None]
    return ssl, spec


def verdict(p: float, det: Detector) -> str:
    """Human-readable decision at the analyst operating point."""
    hi = det.thresholds.get("analyst", 0.5)
    lo = det.thresholds.get("as_coded", 0.5)
    if p >= hi:
        return "synthetic"
    return "uncertain" if p >= lo else "bona fide"


def score_file(
    path: str | Path, model_path: str | None = None, explain: bool = True
) -> ScoreResult:
    """Score one audio file.

    Args:
        path: Audio file in any format ffmpeg or libsndfile can read.
        model_path: Optional detector bundle path.
        explain: Whether to include the evidence breakdown.

    Returns:
        ``probability`` (0 bona fide .. 1 synthetic), ``verdict``,
        ``duration_s``, ``evidence`` and ``error`` (None when decoding worked).

    """
    det = get_detector(model_path)
    name = Path(path).name
    try:
        wav = load_audio(path)
    except (AudioDecodeError, OSError) as err:
        logger.warning("%s: cannot decode (%s); using default score", name, err)
        return ScoreResult(
            file=name,
            probability=det.default_score,
            verdict="undecodable",
            duration_s=0.0,
            ssl_analyzed_duration_s=0.0,
            ssl_truncated=False,
            analysis_coverage={},
            evidence={},
            error=str(err),
        )
    ssl, spec = features_for(wav, det)
    p = float(det.predict(ssl, spec)[0])
    duration = round(wav.size / SAMPLE_RATE, 3)
    ssl_duration = min(duration, MAX_SECONDS)
    return ScoreResult(
        file=name,
        probability=p,
        verdict=verdict(p, det),
        duration_s=duration,
        ssl_analyzed_duration_s=ssl_duration,
        ssl_truncated=wav.size > MAX_SECONDS * SAMPLE_RATE,
        analysis_coverage={
            "ssl": {"start_s": 0.0, "end_s": ssl_duration},
            "spectral": {"start_s": 0.0, "end_s": duration},
        },
        evidence=det.explain(ssl, spec) if explain else {},
        error=None,
    )


def score_directory(
    folder: Path, out_tsv: Path, model_path: str | None = None, threads: int = 8
) -> list[dict]:
    """Score every audio file of a folder and write the HEARSAY TSV.

    Args:
        folder: Folder with the clips.
        out_tsv: Output path (``filename<TAB>cm-score``).
        model_path: Optional detector bundle path.
        threads: Torch intra-op threads.

    Returns:
        One result dictionary per file.

    """
    if threads < 1:
        msg = "threads must be positive"
        raise ValueError(msg)
    files = audio_paths(folder)
    torch.set_num_threads(threads)
    results = []
    for i, p in enumerate(files, 1):
        results.append(score_file(p, model_path, explain=False))
        if i % 100 == 0:
            logger.info("scored %d/%d", i, len(files))
    write_tsv(results, out_tsv)
    return results


def write_tsv(results: list[dict], out_tsv: Path) -> None:
    """Validate and atomically write ``filename<TAB>cm-score`` rows.

    Args:
        results: Dictionaries with ``file`` and ``probability``.
        out_tsv: Destination.

    """
    names = validate_filenames([r["file"] for r in results], source="results")
    values = validate_probabilities([r["probability"] for r in results])
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=out_tsv.parent, delete=False
        ) as fh:
            temp_path = Path(fh.name)
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            writer.writerow(["filename", "cm-score"])
            for name, value in zip(names, values, strict=True):
                writer.writerow([name, f"{float(value):.6f}"])
        temp_path.replace(out_tsv)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    logger.info("wrote %d rows to %s", len(results), out_tsv)
