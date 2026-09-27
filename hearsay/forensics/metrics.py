# Author: Alex Picon <alexnpc@me.com>
"""Evaluation metrics: AUC, EER and the ASVspoof 5 style minimum DCF.

The HackGT scorer is the ASVspoof 5 evaluation code with ``Pspoof = 0.5``,
``Cmiss = 1`` and ``Cfa = 4``. At that default prior,
``minDCF = min_t (Cmiss P_a(t) + Cfa
P_b(t)) / min(Cmiss, Cfa)`` for two error rates whose roles depend on which
class is treated as the target, so both readings are reported:

* ``nsa``: what the brief says, a real clip flagged as synthetic costs 4x
  a missed fake: ``min_t P_miss_fake + 4 P_flag_real``.
* ``code``: the shipped code with bona fide as target and higher score meaning
  bona fide (we pass ``1 - p``): ``min_t P_flag_real + 4 P_miss_fake``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _rates(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_miss_fake, P_flag_real) for every threshold on ``s``.

    A clip is flagged synthetic when its score is strictly above the
    threshold; thresholds run over all unique scores plus the extremes.

    Args:
        y: Labels, 1 synthetic, 0 real.
        s: Synthetic scores (higher means more synthetic).

    Returns:
        Arrays of miss and false-alarm rates.

    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError(
            "labels and finite scores must be matching one-dimensional arrays"
        )
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("metrics require both binary classes")
    y = y.astype(bool)
    thr = np.concatenate([[-np.inf], np.unique(s)])
    fake, real = np.sort(s[y]), np.sort(s[~y])
    miss = np.searchsorted(fake, thr, side="right") / max(len(fake), 1)
    flag = 1 - np.searchsorted(real, thr, side="right") / max(len(real), 1)
    return miss, flag


def min_dcf(
    y: np.ndarray, s: np.ndarray, reading: str = "nsa", *, p_spoof: float = 0.5
) -> float:
    """Return the normalised minimum detection cost.

    Args:
        y: Labels, 1 synthetic, 0 real.
        s: Synthetic scores.
        reading: ``"nsa"`` (false alarm on real costs 4) or ``"code"``.
        p_spoof: Metric prior of synthetic speech; distinct from deployment prior.

    Returns:
        minDCF, 0 is perfect, 1 is no better than a constant answer.

    """
    if not np.isfinite(p_spoof) or not 0 < p_spoof < 1:
        raise ValueError("p_spoof must be finite and strictly between 0 and 1")
    if reading not in {"nsa", "code"}:
        raise ValueError("reading must be 'nsa' or 'code'")
    miss, flag = _rates(y, s)
    a, b = (
        (p_spoof, 4 * (1 - p_spoof)) if reading == "nsa" else (4 * p_spoof, 1 - p_spoof)
    )
    cost = (a * miss + b * flag) / min(a, b)
    return float(min(cost.min(), 1.0))


def eer(y: np.ndarray, s: np.ndarray) -> float:
    """Return the equal error rate.

    Args:
        y: Labels, 1 synthetic, 0 real.
        s: Synthetic scores.

    Returns:
        EER in [0, 1].

    """
    miss, flag = _rates(y, s)
    i = int(np.argmin(np.abs(miss - flag)))
    return float((miss[i] + flag[i]) / 2)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Return ROC AUC (NaN when only one class or no finite scores).

    Args:
        y: Labels, 1 synthetic, 0 real.
        s: Synthetic scores.

    Returns:
        AUC.

    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 4 or len(set(y[ok].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def summary(y: np.ndarray, s: np.ndarray, *, p_spoof: float = 0.5) -> dict[str, float]:
    """Return all metrics for one score vector.

    Args:
        y: Labels, 1 synthetic, 0 real.
        s: Synthetic scores.
        p_spoof: Synthetic-class prior used in the cost calculation.

    Returns:
        AUC, EER and both minDCF readings, rounded.

    """
    return {
        "auc": round(auc(y, s), 4),
        "eer": round(eer(y, s), 4),
        "min_dcf_nsa": round(min_dcf(y, s, "nsa", p_spoof=p_spoof), 4),
        "min_dcf_code": round(min_dcf(y, s, "code", p_spoof=p_spoof), 4),
        "metric_p_spoof": p_spoof,
    }
