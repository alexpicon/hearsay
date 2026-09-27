# Author: Alex Picon <alexnpc@me.com>
"""Shared data structures: the per-expert Finding and the per-file Context."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from forensics.audio import load_audio

if TYPE_CHECKING:
    from forensics.calibration import Calibration

TARGET_SR = 16000


def _clean(value: Any) -> Any:
    """Make a value JSON friendly (NaN becomes None, numpy scalars become floats).

    Args:
        value: Any evidence value.

    Returns:
        A JSON-serialisable version of the value.

    """
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class Finding:
    """What one forensic expert concluded about one file.

    Attributes:
        expert: Expert name (for example ``"spectral"``).
        score: Probability that the clip is synthetic in [0, 1], or None when
            the expert has no opinion.
        confidence: How much the expert trusts its own score, in [0, 1].
        summary: Plain-English explanation of the finding.
        features: Numeric measurements used for scoring.
        evidence: Extra values shown in the report (strings, flags, series).
        contributions: Per-feature log-odds contributions to the score.
        elapsed_ms: Wall time spent by the expert.

    """

    expert: str
    score: float | None
    confidence: float
    summary: str
    features: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self, with_series: bool = False) -> dict[str, Any]:
        """Serialise the finding.

        Args:
            with_series: Keep long plot series (``evidence["series"]``).

        Returns:
            A JSON-friendly dictionary.

        """
        data = asdict(self)
        if not with_series:
            data["evidence"] = {
                k: v for k, v in data["evidence"].items() if k != "series"
            }
        return _clean(data)


@dataclass
class Context:
    """Per-file state shared by the experts and the router.

    Audio is decoded lazily, once, to 16 kHz mono float32. Expensive shared
    intermediate results (voice activity, spectra) live in ``cache``.

    Attributes:
        path: Audio file being analysed.
        calibration: Fitted feature models, or None for uncalibrated runs.
        batch: Batch-level facts gathered by the router (for example the share
            of files whose length matches a trimming grid).
        findings: Findings produced so far, keyed by expert name.
        cache: Scratch space for shared computations.

    """

    path: Path
    calibration: Calibration | None = None
    batch: dict[str, Any] = field(default_factory=dict)
    findings: dict[str, Finding] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    sr: int = TARGET_SR
    _audio: np.ndarray | None = None
    native_sr: int | None = None

    @property
    def audio(self) -> np.ndarray:
        """Return the decoded 16 kHz mono signal, decoding on first use."""
        if self._audio is None:
            self._audio, self.native_sr = load_audio(self.path, self.sr)
        return self._audio

    @property
    def duration(self) -> float:
        """Return the clip duration in seconds."""
        return len(self.audio) / self.sr
