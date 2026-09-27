# Author: Alex Picon <alexnpc@me.com>
"""Evaluation with the organizers' ASVspoof 5 Track 1 scorer.

The official ``calculate_modules.py`` (HackGTMinDCF.tar) is imported when it
is available, otherwise an identical local copy of its two functions is used.
The supplied scorer defaults to Pspoof = 0.5, Cmiss = 1 and Cfa = 4.
The spoof prior is configurable; 0.3 can be reported separately without
silently changing historical evaluations.

Direction matters. The official code treats HIGHER scores as bona fide,
while the HEARSAY answer key asks for 1.0 = synthetic. Two readings exist:

* ``as_coded``: the scorer is fed ``1 - p`` (bona fide high). Then the 4x
  cost falls on spoofs accepted as bona fide (missed fakes).
* ``analyst``: the reading of the brief ("a false alarm, flagging real audio
  as fake, costs 4x"). Spoof is the positive class scored by ``p`` and the 4x
  cost falls on real clips scored above the threshold.

Both readings are reported. ``analyst`` remains the historical headline;
neither reading or prior is asserted to be the final organizer contract.
"""

import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
P_SPOOF, C_MISS, C_FA = 0.5, 1.0, 4.0
SCORER_DIR = Path(
    os.environ.get(
        "HEARSAY_SCORER",
        str(
            Path(__file__).resolve().parents[2]
            / "_internal/hearsay/work/HackGTMinDCF/asvspoof5/evaluation-package"
        ),
    )
)


def _load_official() -> ModuleType | None:
    """Import the organizers' ``calculate_modules.py`` if it exists."""
    path = SCORER_DIR / "calculate_modules.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("hgt_calculate_modules", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL = _load_official()


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    """Require a nonempty one-dimensional vector of finite real numbers."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain finite real numbers")
    try:
        out = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if out.ndim != 1 or not out.size or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
    return out


def _prior(p_spoof: float) -> float:
    """Validate the configured probability of the spoof class."""
    try:
        value = float(p_spoof)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_spoof must be finite and strictly between 0 and 1") from exc
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("p_spoof must be finite and strictly between 0 and 1")
    return value


def det_curve(target: np.ndarray, nontarget: np.ndarray) -> tuple[np.ndarray, ...]:
    """Local copy of the official ``compute_det_curve``.

    Args:
        target: Scores of the positive class (higher = more positive).
        nontarget: Scores of the negative class.

    Returns:
        ``(frr, far, thresholds)``.

    """
    target = _vector(target, "target scores")
    nontarget = _vector(nontarget, "nontarget scores")
    n = target.size + nontarget.size
    scores = np.concatenate((target, nontarget))
    labels = np.concatenate((np.ones(target.size), np.zeros(nontarget.size)))
    idx = np.argsort(scores, kind="mergesort")
    labels = labels[idx]
    tar_sums = np.cumsum(labels)
    non_sums = nontarget.size - (np.arange(1, n + 1) - tar_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_sums / target.size))
    far = np.concatenate((np.atleast_1d(1), non_sums / nontarget.size))
    thr = np.concatenate((np.atleast_1d(scores[idx[0]] - 0.001), scores[idx]))
    return frr, far, thr


def min_dcf(
    frr: np.ndarray,
    far: np.ndarray,
    *,
    p_spoof: float = P_SPOOF,
    target_is_spoof: bool = False,
) -> float:
    """Compute the normalized minimum DCF with the organizers' cost model.

    Args:
        frr: Miss rates of the positive class.
        far: False acceptance rates of the negative class.
        p_spoof: Probability of a synthetic clip, defaulting to the supplied 0.5.
        target_is_spoof: True for the analyst reading; False preserves the
            historical bona-fide-positive convention of this helper.

    Returns:
        The normalized minDCF.

    """
    frr, far = _vector(frr, "frr"), _vector(far, "far")
    if frr.shape != far.shape or np.any((frr < 0) | (frr > 1)):
        raise ValueError("frr and far must have equal shapes and rates in [0, 1]")
    if np.any((far < 0) | (far > 1)):
        raise ValueError("frr and far must have equal shapes and rates in [0, 1]")
    prior = _prior(p_spoof)
    p_target = prior if target_is_spoof else 1 - prior
    c_det = C_MISS * frr * p_target + C_FA * far * (1 - p_target)
    return float(c_det.min() / min(C_MISS * p_target, C_FA * (1 - p_target)))


def _mindcf_eer(
    target: np.ndarray,
    nontarget: np.ndarray,
    *,
    p_spoof: float = P_SPOOF,
    target_is_spoof: bool = False,
) -> tuple[float, float]:
    """MinDCF and EER, through the official code when available."""
    if OFFICIAL is not None:
        eer, frr, far, thr, _ = OFFICIAL.compute_eer(target, nontarget)
        # The supplied function always takes the *nontarget* prior. The analyst
        # target is spoof, so its nontarget prior is the bona fide probability.
        non_prior = 1 - p_spoof if target_is_spoof else p_spoof
        mdcf, _ = OFFICIAL.compute_mindcf(frr, far, thr, non_prior, C_MISS, C_FA)
        return float(mdcf), float(eer)
    frr, far, _ = det_curve(target, nontarget)
    i = int(np.argmin(np.abs(frr - far)))
    return (
        min_dcf(frr, far, p_spoof=p_spoof, target_is_spoof=target_is_spoof),
        float((frr[i] + far[i]) / 2),
    )


def evaluate_scores(
    p_synthetic: np.ndarray, labels: np.ndarray, *, p_spoof: float = P_SPOOF
) -> dict[str, float | int | bool]:
    """All HEARSAY metrics for scores ranked toward the synthetic class.

    Args:
        p_synthetic: Finite scores where higher means more likely synthetic.
            Logits are supported for historical training callers; release TSVs
            are separately required to contain probabilities in [0, 1].
        labels: 1 for synthetic, 0 for bona fide.
        p_spoof: Synthetic-class prior, defaulting to the supplied scorer's 0.5.

    Returns:
        ``min_dcf`` (analyst reading), ``min_dcf_as_coded``, ``eer``, ``auc``
        and class counts, together with the prior and costs used.

    """
    p = _vector(p_synthetic, "scores")
    y = np.asarray(labels)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or np.iscomplexobj(y)
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("labels must be a matching one-dimensional vector of 0 or 1")
    y = y.astype(int)
    prior = _prior(p_spoof)
    spoof, bona = p[y == 1], p[y == 0]
    if not spoof.size or not bona.size:
        raise ValueError("evaluation requires both spoof and bona fide labels")
    analyst, eer = _mindcf_eer(spoof, bona, p_spoof=prior, target_is_spoof=True)
    as_coded, _ = _mindcf_eer(1 - bona, 1 - spoof, p_spoof=prior)
    auc = float(roc_auc_score(y, p))
    return {
        "min_dcf": analyst,
        "min_dcf_as_coded": as_coded,
        "eer": eer,
        "auc": auc,
        "n_spoof": int(spoof.size),
        "n_bonafide": int(bona.size),
        "official_scorer": OFFICIAL is not None,
        "p_spoof": prior,
        "c_miss": C_MISS,
        "c_fa": C_FA,
    }
