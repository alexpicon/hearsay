# Author: Alex Picon <alexnpc@me.com>
"""Training, held-out evaluation and operating-point selection."""

import logging

import numpy as np
from sklearn.model_selection import GroupKFold

from hearsay.backend import Fusion, LinearHead, TreeHead, balanced_weights, sigmoid
from hearsay.metrics import C_FA, C_MISS, evaluate_scores
from hearsay.model import Detector, Features
from hearsay.spectral import feature_names

logger = logging.getLogger(__name__)


SOURCE_SHARE = {"la19": 1.0, "vctk": 1.0, "diffssd": 0.35, "lj": 0.1}


def composition(f: Features) -> dict[str, int]:
    """Return clip counts per corpus and class for evaluation provenance."""
    keys = [
        f"{src}/{'spoof' if lab == 1 else 'bonafide'}"
        for src, lab in zip(f.source, f.label, strict=True)
    ]
    names, counts = np.unique(keys, return_counts=True)
    return {str(k): int(v) for k, v in zip(names, counts, strict=True)}


def training_weights(f: Features) -> np.ndarray:
    """Balance the classes, then corpora and generators within each class.

    Each corpus gets a share of its class (``SOURCE_SHARE``, relative to
    ASVspoof 2019 LA, whose channel and text match the test set) and each
    generator an equal part of its corpus share.

    Args:
        f: Training features.

    Returns:
        Sample weights with mean 1.

    """
    y = (f.label == 1).astype(int)
    w = balanced_weights(y, np.where(y == 1, f.system, f.source))
    for c in (0, 1):
        in_class = y == c
        present = sorted(set(f.source[in_class]))
        shares = {s: SOURCE_SHARE.get(s, 0.2) for s in present}
        total = sum(shares.values())
        mass = w[in_class].sum()
        for s in present:
            m = in_class & (f.source == s)
            w[m] *= shares[s] / total * mass / w[m].sum()
    return w * (w.size / w.sum())


def fit_detector(
    train: Features,
    *,
    ssl_model: str,
    n_layers: int,
    layers: tuple[int, ...],
    c: float = 0.05,
    n_folds: int = 4,
) -> Detector:
    """Fit both heads and learn the fusion on attack-disjoint folds.

    Args:
        train: Training features.
        ssl_model: Front-end id (stored for inference).
        n_layers: Front-end depth (stored for inference).
        layers: SSL hidden states used by the linear head.
        c: Inverse L2 strength of the linear head.
        n_folds: Folds for the out-of-fold fusion training.

    Returns:
        The fitted detector.

    """
    y = (train.label == 1).astype(int)
    w = training_weights(train)
    names = feature_names()
    oof = np.zeros((len(train), 2))
    # Spoofs are grouped by generator and bona fide clips by speaker, so the
    # out-of-fold logits that train the fusion come from unseen attacks.
    groups = np.where(
        y == 1,
        np.char.add("sys:", train.system.astype(str)),
        np.char.add("spk:", train.speaker.astype(str)),
    )
    folds = GroupKFold(n_splits=n_folds).split(train.spec, y, groups=groups)
    for k, (tr, va) in enumerate(folds):
        lin = LinearHead(layers, c).fit(train.ssl[tr], y[tr], w[tr])
        tree = TreeHead(names).fit(train.spec[tr], y[tr], w[tr])
        oof[va, 0] = lin.logit(train.ssl[va])
        oof[va, 1] = tree.logit(train.spec[va])
        logger.info(
            "fold %d: ssl %s | spectral %s",
            k,
            _short(evaluate_scores(oof[va, 0], y[va])),
            _short(evaluate_scores(oof[va, 1], y[va])),
        )
    fusion = Fusion(["ssl_head", "spectral_head"]).fit(oof, y, w)
    logger.info("fusion weights %s", fusion.weights())
    linear = LinearHead(layers, c).fit(train.ssl, y, w)
    tree = TreeHead(names).fit(train.spec, y, w)
    oof_metrics = {
        "ssl_head": evaluate_scores(oof[:, 0], y),
        "spectral_head": evaluate_scores(oof[:, 1], y),
        "fused": evaluate_scores(fusion.logit(oof), y),
        "folds": f"{n_folds} folds, spoofs grouped by generator, bona fide by speaker",
    }
    logger.info("out-of-fold fused %s", _short(oof_metrics["fused"]))
    det = Detector(ssl_model, n_layers, linear, tree, fusion)
    det.thresholds = operating_points(sigmoid(fusion.logit(oof)), y)
    det.default_score = default_score(det.thresholds)
    det.meta = {"oof_attack_disjoint": oof_metrics}
    return det


def _short(m: dict) -> str:
    """Compact metric string for logs."""
    return f"minDCF {m['min_dcf']:.4f} EER {m['eer'] * 100:.2f}% AUC {m['auc']:.4f}"


def operating_points(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Thresholds that achieve minDCF under both scorer readings.

    Args:
        p: Synthetic probabilities.
        y: Labels (1 synthetic).

    Returns:
        ``analyst`` (4x cost on flagging real audio) and ``as_coded``
        (4x cost on missing a fake) thresholds on the probability scale.

    """
    cand = np.unique(np.concatenate([[0.0], p, [1.0 + 1e-9]]))
    real, fake = p[y == 0], p[y == 1]
    fa = np.array([(real >= t).mean() for t in cand])
    miss = np.array([(fake < t).mean() for t in cand])
    analyst = cand[np.argmin(C_MISS * miss + C_FA * fa)]
    as_coded = cand[np.argmin(C_MISS * fa + C_FA * miss)]
    return {"analyst": float(analyst), "as_coded": float(as_coded)}


def default_score(points: dict[str, float]) -> float:
    """Score for undecodable clips, hedged between both scorer readings.

    Under the analyst reading a clip scored below the (high) analyst
    threshold can only become a cheap miss; under the as-coded reading a clip
    above the (low) as-coded threshold can only become a cheap false alarm.
    A score between the two thresholds is the cheap outcome under both.

    Args:
        points: Output of :func:`operating_points`.

    Returns:
        The midpoint of the two thresholds in log-odds space.

    """
    lo, hi = sorted([points["as_coded"], points["analyst"]])
    lo, hi = np.clip([lo, hi], 1e-4, 1 - 1e-4)
    mid = (np.log(lo / (1 - lo)) + np.log(hi / (1 - hi))) / 2
    return float(sigmoid(np.array(mid)))


def evaluate(det: Detector, f: Features) -> dict:
    """Metrics overall and per attack, source and partition.

    Args:
        det: Fitted detector.
        f: Labeled held-out features.

    Returns:
        Nested metric dictionary.

    """
    y = (f.label == 1).astype(int)
    logits = det.head_logits(f.ssl, f.spec)
    p = det.predict(f.ssl, f.spec)
    out = {
        "fused": evaluate_scores(p, y),
        "ssl_head": evaluate_scores(logits[:, 0], y),
        "spectral_head": evaluate_scores(logits[:, 1], y),
        "operating_points": operating_points(p, y),
        "per_system": {},
    }
    bona = y == 0
    for s in sorted(set(f.system[y == 1])):
        m = bona | (f.system == s)
        out["per_system"][s] = {
            k: round(v, 4)
            for k, v in evaluate_scores(p[m], y[m]).items()
            if k in ("min_dcf", "eer", "auc")
        }
    logger.info("fused %s", _short(out["fused"]))
    return out
