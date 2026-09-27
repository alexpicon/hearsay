# Author: Alex Picon <alexnpc@me.com>
"""Small, inspectable per-expert models that turn measurements into scores.

Every expert measures a handful of named features. A standardised logistic
model per expert (a few coefficients stored in JSON) maps them to a synthetic
probability, and robust statistics of real speech let the report say *how*
unusual each value is. Nothing here is a black box: the log-odds contribution
of every feature is returned alongside the score.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).with_name("calibration.json")


def sigmoid(x: float) -> float:
    """Return the logistic function of ``x``.

    Args:
        x: Log-odds.

    Returns:
        Probability in (0, 1).

    """
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def logit(p: float, eps: float = 1e-4) -> float:
    """Return the log-odds of a probability, clipped away from 0 and 1.

    Args:
        p: Probability.
        eps: Clipping margin.

    Returns:
        Log-odds.

    """
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


@dataclass
class FeatureModel:
    """Standardised logistic model for one expert.

    Attributes:
        features: Feature names, in coefficient order.
        mean: Training mean per feature (also the NaN fill value).
        scale: Training standard deviation per feature.
        coef: Logistic coefficients on standardised features.
        intercept: Logistic intercept.
        real_median: Median of each feature on real (bona fide) speech.
        real_spread: Robust spread (IQR / 1.349) on real speech.
        cv_auc: Cross-validated AUC measured when the model was fitted.

    """

    features: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    real_median: list[float]
    real_spread: list[float]
    cv_auc: float | None = None

    def score(self, feats: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Score a feature dictionary.

        Args:
            feats: Measured features (missing or NaN values count as average).

        Returns:
            Synthetic probability and the log-odds contribution per feature.

        """
        contrib: dict[str, float] = {}
        for i, name in enumerate(self.features):
            value = feats.get(name)
            if value is None or not math.isfinite(value):
                continue
            z = (value - self.mean[i]) / (self.scale[i] or 1.0)
            contrib[name] = self.coef[i] * max(-6.0, min(6.0, z))
        return sigmoid(self.intercept + sum(contrib.values())), contrib

    def real_z(self, name: str, value: float) -> float:
        """Return how many robust standard deviations ``value`` is from real.

        Args:
            name: Feature name.
            value: Measured value.

        Returns:
            Signed robust z-score relative to real speech.

        """
        i = self.features.index(name)
        return (value - self.real_median[i]) / (self.real_spread[i] or 1.0)


@dataclass
class Calibration:
    """All fitted models plus the fusion weights.

    Attributes:
        experts: Per-expert feature models.
        fusion_weights: Weight on each expert's log-odds in the fused score.
        fusion_bias: Bias of the fused log-odds.
        meta: Provenance (training sets, sizes, date).

    """

    experts: dict[str, FeatureModel] = field(default_factory=dict)
    fusion_weights: dict[str, float] = field(default_factory=dict)
    fusion_bias: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> Calibration | None:
        """Load a calibration file.

        Args:
            path: JSON file written by :meth:`save`.

        Returns:
            The calibration, or None if the file does not exist.

        """
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        experts = {k: FeatureModel(**v) for k, v in raw["experts"].items()}
        return cls(
            experts=experts,
            fusion_weights=raw.get("fusion_weights", {}),
            fusion_bias=raw.get("fusion_bias", 0.0),
            meta=raw.get("meta", {}),
        )

    def save(self, path: Path = DEFAULT_PATH) -> None:
        """Write the calibration as indented JSON.

        Args:
            path: Destination file.

        """
        data = {
            "experts": {k: asdict(v) for k, v in self.experts.items()},
            "fusion_weights": self.fusion_weights,
            "fusion_bias": self.fusion_bias,
            "meta": self.meta,
        }
        path.write_text(json.dumps(data, indent=1))


def explain(
    model: FeatureModel | None,
    feats: dict[str, float],
    contrib: dict[str, float],
    descriptions: dict[str, str],
    top: int = 2,
) -> str:
    """Write a one-line explanation from the strongest feature contributions.

    Args:
        model: Fitted model (for the real-speech reference values).
        feats: Measured features.
        contrib: Log-odds contributions from :meth:`FeatureModel.score`.
        descriptions: Plain-English name per feature.
        top: Number of features to mention.

    Returns:
        Explanation text.

    """
    if not contrib or model is None:
        return "no calibrated evidence"
    ranked = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:top]
    parts = []
    for name, c in ranked:
        value = feats[name]
        ref = model.real_median[model.features.index(name)]
        z = model.real_z(name, value)
        side = "synthetic" if c > 0 else "real"
        label = descriptions.get(name, name)
        parts.append(
            f"{label} = {value:.3g} (real median {ref:.3g}, z={z:+.1f}) "
            f"points to {side}"
        )
    return "; ".join(parts)
