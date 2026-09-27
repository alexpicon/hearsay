# Author: Alex Picon <alexnpc@me.com>
"""Light, inspectable classifiers on top of the cached features.

* ``LinearHead``: standardized SSL layer statistics into an L2 logistic
  regression. Linear, so each layer's share of the logit can be reported.
* ``TreeHead``: LightGBM on the named hand-crafted features. Its native
  SHAP contributions explain every decision in acoustic terms.
* ``Fusion``: a logistic regression over the heads' out-of-fold logits.
"""

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
SHORTCUT_FEATURES = ("duration_s", "lead_silence_s", "trail_silence_s")


def ssl_matrix(ssl: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    """Flatten the chosen layers' mean and std statistics.

    Args:
        ssl: Array ``(N, n_layers + 1, 2, dim)``.
        layers: Hidden-state indices to keep (0 is the CNN output).

    Returns:
        Array ``(N, len(layers) * 2 * dim)`` in float32.

    """
    return ssl[:, list(layers)].reshape(ssl.shape[0], -1).astype(np.float32)


def balanced_weights(y: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Sample weights that balance classes (and groups within a class).

    Args:
        y: Binary labels.
        groups: Optional group ids (for example generator names) so that a
            generator with many clips does not dominate its class.

    Returns:
        Weights with mean 1.

    """
    w = np.ones(y.size, dtype=np.float64)
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            continue
        if groups is None:
            w[idx] = 1.0 / idx.size
            continue
        g = groups[idx]
        names, counts = np.unique(g, return_counts=True)
        per = dict(zip(names, counts, strict=True))
        w[idx] = np.array([1.0 / (len(names) * per[k]) for k in g])
    return w * (w.size / w.sum())


@dataclass
class LinearHead:
    """Standard scaler plus L2 logistic regression on SSL statistics."""

    layers: tuple[int, ...]
    c: float = 0.05
    scaler: StandardScaler = field(default_factory=StandardScaler)
    clf: LogisticRegression | None = None

    def fit(self, ssl: np.ndarray, y: np.ndarray, w: np.ndarray) -> LinearHead:
        """Fit on SSL features.

        Args:
            ssl: SSL array ``(N, L+1, 2, D)``.
            y: Labels (1 synthetic).
            w: Sample weights.

        Returns:
            Self.

        """
        x = self.scaler.fit_transform(ssl_matrix(ssl, self.layers))
        self.clf = LogisticRegression(C=self.c, max_iter=3000)
        self.clf.fit(x, y, sample_weight=w)
        return self

    def logit(self, ssl: np.ndarray) -> np.ndarray:
        """Decision value (log-odds of synthetic)."""
        x = self.scaler.transform(ssl_matrix(ssl, self.layers))
        return self.clf.decision_function(x)

    def layer_contributions(self, ssl: np.ndarray) -> np.ndarray:
        """Per-layer share of the logit, shape ``(N, len(layers))``."""
        x = self.scaler.transform(ssl_matrix(ssl, self.layers))
        parts = (x * self.clf.coef_[0]).reshape(x.shape[0], len(self.layers), -1)
        return parts.sum(axis=2)


@dataclass
class TreeHead:
    """LightGBM on hand-crafted features with SHAP explanations.

    Duration and leading/trailing silence are excluded on purpose: in
    ASVspoof 2019 LA they separate bona fide from spoofed clips for reasons
    unrelated to synthesis (a known dataset shortcut), and the test clips
    were selected to be at least 3 s long.
    """

    names: list[str]
    exclude: tuple[str, ...] = SHORTCUT_FEATURES
    params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
            "verbose": -1,
        }
    )
    model: lgb.LGBMClassifier | None = None

    @property
    def columns(self) -> np.ndarray:
        """Indices of the features the model uses."""
        return np.array([i for i, n in enumerate(self.names) if n not in self.exclude])

    def fit(self, spec: np.ndarray, y: np.ndarray, w: np.ndarray) -> TreeHead:
        """Fit on hand-crafted features.

        Args:
            spec: Feature matrix aligned with ``names``.
            y: Labels (1 synthetic).
            w: Sample weights.

        Returns:
            Self.

        """
        self.model = lgb.LGBMClassifier(**self.params, n_jobs=8)
        self.model.fit(spec[:, self.columns], y, sample_weight=w)
        return self

    def logit(self, spec: np.ndarray) -> np.ndarray:
        """Decision value (log-odds of synthetic)."""
        return self.model.predict(spec[:, self.columns], raw_score=True)

    def contributions(self, spec: np.ndarray) -> np.ndarray:
        """SHAP contributions ``(N, n_features + 1)``; last column is bias.

        Excluded features get a contribution of exactly zero.
        """
        raw = self.model.predict(spec[:, self.columns], pred_contrib=True)
        out = np.zeros((spec.shape[0], len(self.names) + 1))
        out[:, self.columns] = raw[:, :-1]
        out[:, -1] = raw[:, -1]
        return out


@dataclass
class Fusion:
    """Logistic-regression fusion of head logits with non-negative weights.

    Weights are constrained to be >= 0 so that every head can only add
    evidence in the direction it was trained for, which keeps the fused score
    explainable ("both heads point to synthetic") and stops the fusion from
    exploiting correlations between heads that do not transfer.
    """

    names: list[str]
    coef: np.ndarray | None = None
    bias: float = 0.0

    def fit(self, logits: np.ndarray, y: np.ndarray, w: np.ndarray) -> Fusion:
        """Fit fusion weights on out-of-fold logits.

        Args:
            logits: Array ``(N, n_heads)``.
            y: Labels.
            w: Sample weights.

        Returns:
            Self.

        """
        n = logits.shape[1]
        t = 2.0 * y - 1.0

        def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
            z = logits @ theta[:n] + theta[n]
            m = -t * z
            val = (
                np.sum(w * np.logaddexp(0.0, m)) / w.sum()
                + 1e-3 * theta[:n] @ theta[:n]
            )
            g = -t * w * sigmoid(m) / w.sum()
            grad = np.append(logits.T @ g + 2e-3 * theta[:n], g.sum())
            return float(val), grad

        bounds = [(0.0, None)] * n + [(None, None)]
        res = minimize(
            loss,
            np.append(np.ones(n) / n, 0.0),
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
        )
        self.coef, self.bias = res.x[:n], float(res.x[n])
        return self

    def logit(self, logits: np.ndarray) -> np.ndarray:
        """Fused log-odds."""
        return logits @ self.coef + self.bias

    def weights(self) -> dict[str, float]:
        """Fusion weight per head."""
        return dict(zip(self.names, self.coef.round(4).tolist(), strict=True))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically safe logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
