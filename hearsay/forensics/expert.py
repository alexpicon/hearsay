# Author: Alex Picon <alexnpc@me.com>
"""Glue shared by all experts: calibrated scoring and timing."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from forensics.calibration import explain
from forensics.finding import Context, Finding

Extractor = Callable[[Context], tuple[dict[str, float], dict[str, Any]]]


def build_finding(
    ctx: Context,
    expert: str,
    feats: dict[str, float],
    evidence: dict[str, Any],
    descriptions: dict[str, str],
    confidence: float,
    started: float,
    note: str = "",
) -> Finding:
    """Turn raw measurements into a scored, explained Finding.

    When the calibration has a model for the expert, the score is that
    model's probability and the summary names the features that moved it
    most. Without a model the expert reports its measurements with no score.

    Args:
        ctx: File context (holds the calibration).
        expert: Expert name.
        feats: Numeric features.
        evidence: Extra report values.
        descriptions: Plain-English feature names.
        confidence: Expert self-assessed confidence in [0, 1].
        started: ``time.perf_counter()`` value when the expert started.
        note: Extra sentence prepended to the summary.

    Returns:
        The finding.

    """
    model = ctx.calibration.experts.get(expert) if ctx.calibration else None
    if expert == "speaker" and evidence.get("backend") != "wavlm-xvector":
        model = None
        confidence = 0.0
        evidence["calibration_status"] = "uncalibrated_backend"
        note = "Speaker measurements are uncalibrated for this backend; abstaining."
    finite = {k: v for k, v in feats.items() if v is not None and math.isfinite(v)}
    score: float | None = None
    contrib: dict[str, float] = {}
    if model is not None and finite:
        score, contrib = model.score(finite)
    reason = explain(model, finite, contrib, descriptions)
    summary = f"{note} {reason}".strip() if note else reason
    return Finding(
        expert=expert,
        score=score,
        confidence=max(0.0, min(1.0, confidence)),
        summary=summary,
        features=feats,
        evidence=evidence,
        contributions=contrib,
        elapsed_ms=1000 * (time.perf_counter() - started),
    )


def run_extractor(
    ctx: Context,
    expert: str,
    extractor: Extractor,
    descriptions: dict[str, str],
    confidence: Callable[[Context, dict[str, float]], float],
) -> Finding:
    """Run a feature extractor and wrap the result as a Finding.

    Args:
        ctx: File context.
        expert: Expert name.
        extractor: Function returning (features, evidence).
        descriptions: Plain-English feature names.
        confidence: Function computing the expert's confidence.

    Returns:
        The finding.

    """
    started = time.perf_counter()
    feats, evidence = extractor(ctx)
    note = str(evidence.pop("note", ""))
    return build_finding(
        ctx,
        expert,
        feats,
        evidence,
        descriptions,
        confidence(ctx, feats),
        started,
        note,
    )
