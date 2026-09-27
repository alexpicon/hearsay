# Author: Alex Picon <alexnpc@me.com>
"""Nested held-out expert/fusion evaluation and cross-condition diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from forensics.fit import FITTED, fit_calibration, matrix, rescore
from forensics.fusion import fuse
from forensics.metrics import auc, summary
from forensics.validation import PROTOCOL, calibration_folds

Rows = list[dict[str, Any]]


def single_feature_aucs(rows: Rows, y: np.ndarray, expert: str) -> dict[str, float]:
    """Describe feature-label associations; these are not model validation."""
    x = matrix(rows, expert)
    return {n: round(auc(y, x[:, j]), 3) for j, n in enumerate(FITTED[expert])}


def held_out_scores(
    rows: Rows,
    y: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], int]:
    """Score each clip only with experts and fusion trained outside its fold.

    Omission scores remove one expert at inference from the same held-out
    prediction. They do not refit the other experts or the fusion.
    """
    names = [*FITTED, "container"]
    scores = {name: np.full(len(y), np.nan) for name in names}
    fused = np.empty(len(y))
    omitted = {name: np.empty(len(y)) for name in names}
    count = 0
    for held, cal in calibration_folds(rows, y):
        count += 1
        for i in held:
            found = {k: v for k, v in rescore(rows[i], cal).items() if k in names}
            fused[i] = fuse(found, cal, 0.5)[0]
            for name in names:
                finding = found.get(name)
                if finding is not None and finding.score is not None:
                    scores[name][i] = finding.score
                without = {k: v for k, v in found.items() if k != name}
                omitted[name][i] = fuse(without, cal, 0.5)[0]
    return scores, fused, omitted, count


def within(
    rows: Rows, y: np.ndarray, groups: list[str], *, p_spoof: float = 0.5
) -> dict[str, Any]:
    """Evaluate nested clip-disjoint CV; group names label diagnostic slices.

    These folds hold out source paths, not whole generators or speakers.
    ``groups`` is used only for the per-generator metric breakdown.
    """
    y = np.asarray(y)
    if len(groups) != len(rows):
        raise ValueError("one diagnostic group is required per analysis row")
    scores, fused, omitted, count = held_out_scores(rows, y)
    out: dict[str, Any] = {
        "validation": {
            "protocol": PROTOCOL,
            "outer_folds": count,
            "metric_p_spoof": p_spoof,
            "ablation": "held-out inference omission; fusion is not refitted",
        },
        "experts": {},
        "fused": summary(y, fused, p_spoof=p_spoof),
    }
    for name, values in scores.items():
        out["experts"][name] = summary(
            y, np.nan_to_num(values, nan=0.5), p_spoof=p_spoof
        ) | {"coverage": round(float(np.isfinite(values).mean()), 3)}
        if name in FITTED:
            out["experts"][name]["feature_auc_descriptive"] = single_feature_aucs(
                rows, y, name
            )
    out["heldout_expert_omission_auc_drop"] = {
        name: round(auc(y, fused) - auc(y, values), 4)
        for name, values in omitted.items()
    }
    g = np.asarray(groups)
    out["fused_auc_by_group"] = {
        name: round(auc(y[(y == 0) | (g == name)], fused[(y == 0) | (g == name)]), 3)
        for name in sorted(set(g[y == 1].tolist()))
    }
    return out


def transfer(
    train: tuple[Rows, np.ndarray],
    test: tuple[Rows, np.ndarray],
    label: str,
    *,
    p_spoof: float = 0.5,
) -> dict[str, Any]:
    """Fit one dataset and score another; shared-source transforms are diagnostics.

    Dataset membership alone does not establish source-clip independence.
    In particular, clean and channel-transformed copies can share speech.
    """
    cal = fit_calibration(*train, {"train": label})
    rows_b, y_b = test
    per: dict[str, list[float]] = {}
    probs = []
    for row in rows_b:
        found = {
            k: v for k, v in rescore(row, cal).items() if k in {*FITTED, "container"}
        }
        probs.append(fuse(found, cal, 0.5)[0])
        for name in [*FITTED, "container"]:
            f = found.get(name)
            score = np.nan if f is None or f.score is None else f.score
            per.setdefault(name, []).append(score)
    return {
        "validation": "cross-dataset diagnostic; source independence not asserted",
        "metric_p_spoof": p_spoof,
        "experts": {
            k: round(auc(y_b, np.nan_to_num(np.array(v, dtype=float), nan=0.5)), 4)
            for k, v in per.items()
        },
        "fused": summary(y_b, np.array(probs), p_spoof=p_spoof),
    }
