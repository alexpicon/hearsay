# Author: Alex Picon <alexnpc@me.com>
"""Tests for the classifier heads and the non-negative fusion."""

import numpy as np

from hearsay.backend import Fusion, LinearHead, TreeHead, balanced_weights, sigmoid
from hearsay.spectral import feature_names


def _toy(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return toy SSL statistics, hand-crafted features and labels."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    ssl = rng.standard_normal((n, 4, 2, 8)).astype(np.float16)
    ssl[:, 2, 0, 0] += 2.0 * y
    spec = rng.standard_normal((n, len(feature_names()))).astype(np.float32)
    spec[:, 0] += 1.5 * y
    return ssl, spec, y


class TestHeads:
    """The heads learn a planted signal and explain it."""

    def test_linear_head_separates(self) -> None:
        """Planted SSL signal gives high training AUC and layer attribution."""
        ssl, _, y = _toy()
        head = LinearHead((1, 2), c=1.0).fit(ssl, y, np.ones(y.size))
        z = head.logit(ssl)
        assert z[y == 1].mean() > z[y == 0].mean() + 1
        contrib = head.layer_contributions(ssl)
        assert contrib.shape == (y.size, 2)
        np.testing.assert_allclose(
            contrib.sum(axis=1) + head.clf.intercept_[0], z, rtol=1e-4, atol=1e-4
        )

    def test_tree_head_ignores_shortcuts(self) -> None:
        """Excluded features get zero SHAP contribution."""
        _, spec, y = _toy()
        head = TreeHead(feature_names()).fit(spec, y, np.ones(y.size))
        contrib = head.contributions(spec[:5])
        names = feature_names()
        for n in head.exclude:
            assert np.all(contrib[:, names.index(n)] == 0)
        np.testing.assert_allclose(contrib.sum(axis=1), head.logit(spec[:5]), atol=1e-6)


class TestFusion:
    """Non-negative logistic fusion."""

    def test_weights_are_non_negative(self) -> None:
        """An anti-correlated head gets weight 0 instead of a negative one."""
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.5).astype(int)
        good = 3 * (y - 0.5) + rng.standard_normal(500)
        bad = -good + 0.1 * rng.standard_normal(500)
        fusion = Fusion(["good", "bad"]).fit(
            np.column_stack([good, bad]), y, np.ones(500)
        )
        w = fusion.weights()
        assert w["good"] > 0
        assert w["bad"] == 0.0

    def test_balanced_weights(self) -> None:
        """Both classes and every group carry equal total weight."""
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        g = np.array(["a", "a", "b", "x", "x", "x", "y", "y"])
        w = balanced_weights(y, g)
        assert np.isclose(w[y == 0].sum(), w[y == 1].sum())
        assert np.isclose(w[g == "x"].sum(), w[g == "y"].sum())
        assert np.isclose(w.mean(), 1.0)

    def test_sigmoid_is_stable(self) -> None:
        """Extreme logits do not overflow."""
        out = sigmoid(np.array([-1e6, 0.0, 1e6]))
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-12)
