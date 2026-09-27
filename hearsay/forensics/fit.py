# Author: Alex Picon <alexnpc@me.com>
"""Fit the per-expert feature models and the fusion weights."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from forensics import compression, environment, prosody, speaker, spectral, splice
from forensics.calibration import Calibration, FeatureModel, logit
from forensics.finding import Finding
from forensics.metrics import auc
from forensics.splits import folds

FITTED = {
    "spectral": list(spectral.DESCRIPTIONS),
    "compression": list(compression.DESCRIPTIONS),
    "prosody": list(prosody.DESCRIPTIONS),
    "environment": list(environment.DESCRIPTIONS),
    "splice": list(splice.DESCRIPTIONS),
    "speaker": list(speaker.DESCRIPTIONS),
}
Row = dict[str, Any]
# The container expert is rule based; its weight is fixed rather than learned
# so the fusion cannot learn corpus provenance from file headers.
CONTAINER_WEIGHT = 1.0
# The deep detector's probability is already calibrated; until out-of-fold
# deep scores exist for the labelled sets, its log-odds enter with weight 1.
DEEP_WEIGHT = 1.0


def matrix(rows: list[Row], expert: str) -> np.ndarray:
    """Stack one expert's features (NaN where missing).

    Args:
        rows: Cached analyses.
        expert: Expert name.

    Returns:
        Array shaped (files, features).

    """
    names = FITTED[expert]
    out = np.full((len(rows), len(names)), np.nan)
    for i, row in enumerate(rows):
        finding = row.get("findings", {}).get(expert, {})
        if (
            expert == "speaker"
            and finding.get("evidence", {}).get("backend") != "wavlm-xvector"
        ):
            continue
        feats = finding.get("features", {})
        for j, name in enumerate(names):
            v = feats.get(name)
            out[i, j] = v if v is not None and math.isfinite(v) else np.nan
    return out


def fit_model(x: np.ndarray, y: np.ndarray, names: list[str]) -> FeatureModel:
    """Fit a standardised, L2-regularised logistic model.

    Args:
        x: Features (NaN allowed).
        y: Labels, 1 synthetic.
        names: Feature names.

    Returns:
        The fitted model.

    """
    mean = np.nanmean(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.nanstd(x, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    z = np.clip((np.where(np.isfinite(x), x, mean) - mean) / scale, -6, 6)
    clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")
    clf.fit(z, y)
    real = x[y == 0]
    med = np.nanmedian(real, axis=0)
    q75, q25 = np.nanpercentile(real, [75, 25], axis=0)
    spread = np.where((q75 - q25) > 1e-9, (q75 - q25) / 1.349, scale)
    return FeatureModel(
        features=names,
        mean=mean.tolist(),
        scale=scale.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
        real_median=np.where(np.isfinite(med), med, mean).tolist(),
        real_spread=np.where(np.isfinite(spread), spread, 1.0).tolist(),
    )


def oof(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Return 5-fold out-of-fold probabilities (NaN rows stay NaN).

    Args:
        x: Features.
        y: Labels.
        names: Feature names.
        seed: Fold shuffling seed.
        groups: Source identities; repeated clips stay in one fold.

    Returns:
        Out-of-fold synthetic probabilities.

    """
    covered = np.isfinite(x).any(axis=1)
    out = np.full(len(y), np.nan)
    idx = np.flatnonzero(covered)
    if len(set(y[idx].tolist())) < 2 or len(idx) < 20:
        return out
    try:
        split = folds(y[idx], None if groups is None else groups[idx], seed=seed)
    except ValueError:
        return out
    for tr, te in split:
        model = fit_model(x[idx[tr]], y[idx[tr]], names)
        out[idx[te]] = [
            model.score(dict(zip(names, row, strict=True)))[0] for row in x[idx[te]]
        ]
    return out


def expert_logits(
    rows: list[Row],
    scores: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Build the fusion design matrix: confidence-weighted expert log-odds.

    Args:
        rows: Cached analyses (for confidences and the container score).
        scores: Per-expert probability vectors (NaN when absent).

    Returns:
        Matrix (files, experts) with 0 where an expert gave no score, and the
        expert order.

    """
    names = sorted(scores)
    out = np.zeros((len(rows), len(names)))
    for j, name in enumerate(names):
        for i, row in enumerate(rows):
            p = scores[name][i]
            conf = row.get("findings", {}).get(name, {}).get("confidence", 0.0) or 0.0
            if np.isfinite(p):
                out[i, j] = conf * logit(float(p))
    return out, names


def container_scores(rows: list[Row]) -> np.ndarray:
    """Return the rule-based container probabilities (NaN when no opinion).

    Args:
        rows: Cached analyses.

    Returns:
        Probabilities.

    """
    out = []
    for row in rows:
        s = row.get("findings", {}).get("container", {}).get("score")
        out.append(np.nan if s is None else s)
    return np.array(out, dtype=float)


def fit_calibration(
    rows: list[Row],
    y: np.ndarray,
    note: dict[str, Any],
) -> Calibration:
    """Fit every expert model and the fusion weights on labelled rows.

    Fusion weights come from a logistic regression on out-of-fold expert
    log-odds, so the weights are not fooled by in-sample overconfidence.

    Args:
        rows: Cached analyses.
        y: Labels, 1 synthetic.
        note: Provenance stored in ``meta``.

    Returns:
        The calibration.

    """
    cal = Calibration(meta=dict(note))
    groups = np.array([str(r.get("path", i)) for i, r in enumerate(rows)])
    scores = {"container": container_scores(rows)}
    aucs: dict[str, float] = {}
    for expert, names in FITTED.items():
        x = matrix(rows, expert)
        if np.isfinite(x).any(axis=1).sum() < 20:
            continue
        covered = np.isfinite(x).any(axis=1)
        if len(np.unique(y[covered])) < 2:
            continue
        cal.experts[expert] = fit_model(x[covered], y[covered], names)
        scores[expert] = oof(x, y, names, groups=groups)
        aucs[expert] = auc(y, scores[expert])
        cal.experts[expert].cv_auc = aucs[expert]
    learned = {k: v for k, v in scores.items() if k != "container"}
    design, order = expert_logits(rows, learned)
    fuser = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    fuser.fit(design if order else np.zeros((len(y), 1)), y)
    weights = fuser.coef_[0].round(4).tolist() if order else []
    cal.fusion_weights = dict(zip(order, weights, strict=True))
    cal.fusion_weights["container"] = CONTAINER_WEIGHT
    cal.fusion_weights["deep"] = DEEP_WEIGHT
    cal.fusion_bias = float(fuser.intercept_[0])
    cal.meta["expert_auc"] = {k: round(v, 4) for k, v in aucs.items()}
    cal.meta["fitted"] = dt.datetime.now().isoformat(timespec="seconds")
    return cal


def rescore(row: Row, cal: Calibration) -> dict[str, Finding]:
    """Recalibrate cached measurements, removing incompatible legacy scores."""
    out: dict[str, Finding] = {}
    for name, data in row.get("findings", {}).items():
        finding = Finding(
            **{k: v for k, v in data.items() if k in Finding.__dataclass_fields__}
        )
        finding.evidence = dict(finding.evidence)
        if name == "environment" and finding.evidence.pop("rule_logodds", None):
            finding.summary = (
                "Near-zero segments are processing evidence, not proof of synthesis."
            )
        model = cal.experts.get(name)
        if name in FITTED:
            finding.score, finding.contributions = None, {}
        if name == "speaker" and finding.evidence.get("backend") != "wavlm-xvector":
            model, finding.confidence = None, 0.0
            finding.evidence["calibration_status"] = "uncalibrated_backend"
            finding.summary = (
                "Speaker measurements are uncalibrated for this backend; abstaining."
            )
        feats = {k: v for k, v in finding.features.items() if v is not None}
        if model is not None and any(math.isfinite(v) for v in feats.values()):
            finding.score, finding.contributions = model.score(feats)
        out[name] = finding
    return out
