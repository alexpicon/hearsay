# Author: Alex Picon <alexnpc@me.com>
"""Adapter that lets the router consult the deep-learning anti-spoofing model.

The deep detector lives in the ``hearsay`` package. The router treats it as
one more expert, but a black-box one, so by default it is consulted only
when the explainable experts are uncertain. Scores can come from a
precomputed TSV (``filename<TAB>cm-score``, set ``HEARSAY_DEEP_SCORES``) or
from a ``score_file(path) -> float`` function if the package exposes one.
"""

from __future__ import annotations

import csv
import functools
import importlib
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from forensics.finding import Context, Finding

log = logging.getLogger(__name__)
CANDIDATE_MODULES = ("hearsay.inference", "hearsay.infer", "hearsay.predict")


@functools.cache
def _table(path: str) -> dict[str, float]:
    """Load a precomputed score table.

    Args:
        path: TSV with ``filename`` and ``cm-score`` columns.

    Returns:
        Mapping from file name to synthetic probability.

    """
    with open(path, encoding="utf-8", newline="") as fh:
        rows = csv.DictReader(fh, delimiter="\t")
        return {r["filename"]: float(r["cm-score"]) for r in rows}


@functools.cache
def _scorer() -> Callable[[Path], float] | None:
    """Find a ``score_file`` function in the deep-detector package.

    Returns:
        The function, or None if the package does not expose one.

    """
    for name in CANDIDATE_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(module, "score_file", None)
        if callable(fn):
            return fn
    return None


def _model_ready() -> bool:
    """Return True when the deep detector's trained bundle is on disk.

    Returns:
        Whether a model file exists (True when the layout is unknown).

    """
    configured = os.environ.get("HEARSAY_MODEL")
    if configured:
        return Path(configured).exists()
    try:
        from hearsay.model import MODEL_PATH
    except ImportError:
        return True
    return Path(MODEL_PATH).exists()


def available() -> bool:
    """Return True when a deep score source is configured.

    Returns:
        Whether the deep expert can run.

    """
    table = os.environ.get("HEARSAY_DEEP_SCORES")
    if table and Path(table).exists():
        return True
    return _scorer() is not None and _model_ready()


def analyze(path: Path, ctx: Context) -> Finding:
    """Fetch the deep detector's score for one file.

    Args:
        path: Audio file.
        ctx: File context.

    Returns:
        The deep-detector finding (score None when unavailable).

    """
    started = time.perf_counter()
    table = os.environ.get("HEARSAY_DEEP_SCORES")
    score: float | None = None
    source = "unavailable"
    if table and Path(table).exists():
        score = _table(table).get(path.name)
        source = f"precomputed table {Path(table).name}"
    extra: dict[str, object] = {}
    if score is None and (fn := _scorer()) is not None and _model_ready():
        result = fn(path)
        score = float(result)
        source = f"{fn.__module__}.score_file"
        if isinstance(result, dict):
            extra = {k: result[k] for k in ("verdict", "evidence") if k in result}
    summary = (
        f"Deep anti-spoofing model ({source}) gives {score:.3f}."
        if score is not None
        else "Deep anti-spoofing model not available."
    )
    return Finding(
        expert="deep",
        score=score,
        confidence=0.0 if score is None else 1.0,
        summary=summary,
        features={} if score is None else {"deep_score": score},
        evidence={"source": source, "black_box": True} | extra,
        elapsed_ms=1000 * (time.perf_counter() - started),
    )
