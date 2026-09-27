# Author: Alex Picon <alexnpc@me.com>
"""Explanations describe head scores without contradicting the saved decision."""

from types import SimpleNamespace

import numpy as np
import pytest

from hearsay.backend import Fusion
from hearsay.explain import summarize
from hearsay.inference import verdict
from hearsay.model import Detector


def _detector(probability: float, spectral_weight: float = 0.0) -> Detector:
    """Make deterministic head fixtures without loading or changing saved models."""
    linear = SimpleNamespace(
        layers=(3,),
        logit=lambda _: np.array([0.0]),
        layer_contributions=lambda _: np.array([[0.75]]),
    )
    tree = SimpleNamespace(
        names=["flatness_mean"],
        logit=lambda _: np.array([4.0]),
        contributions=lambda _: np.array([[0.5, 3.5]]),
    )
    fusion = Fusion(
        ["ssl_head", "spectral_head"],
        coef=np.array([1.0, spectral_weight]),
        bias=float(np.log(probability / (1 - probability))),
    )
    return Detector(
        "fixture",
        1,
        linear,
        tree,
        fusion,
        thresholds={"analyst": 0.3, "as_coded": 0.1},
    )


@pytest.mark.parametrize(
    ("score", "decision"), [(0.2, "uncertain"), (0.4, "synthetic")]
)
def test_summary_does_not_assign_a_second_verdict(score: float, decision: str) -> None:
    """Sub-0.5 scores can be uncertain or synthetic under saved operating points."""
    detector = _detector(score)
    ssl, spectral = np.zeros((1, 1)), np.zeros((1, 1))
    before = detector.predict(ssl, spectral)
    evidence = detector.explain(ssl, spectral)
    after = detector.predict(ssl, spectral)
    np.testing.assert_array_equal(before, after)
    assert verdict(float(before[0]), detector) == decision
    assert evidence["summary"].startswith(f"Fused synthetic score: {score:.2f}.")
    assert "bona fide" not in evidence["summary"]
    assert "points to" not in evidence["summary"]
    assert "agrees" not in evidence["summary"]
    assert "SSL head synthetic score: 0.50" in evidence["summary"]
    assert "Spectral head synthetic score: 0.98" in evidence["summary"]
    assert (
        "diagnostic only with zero influence on the fused score" in evidence["summary"]
    )
    np.testing.assert_array_equal(detector.fusion.coef, [1.0, 0.0])


def test_small_nonzero_weight_is_not_described_as_zero() -> None:
    """Human wording uses exact weights rather than rounded display metadata."""
    detector = _detector(0.4, spectral_weight=0.00001)
    evidence = detector.explain(np.zeros((1, 1)), np.zeros((1, 1)))
    assert evidence["fusion_weights"]["spectral_head"] == 0.0
    assert "zero influence" not in evidence["summary"]
    assert "logit fusion weight 1e-05" in evidence["summary"]


def test_summary_without_weights_makes_no_influence_claim() -> None:
    """Legacy helper callers can omit weights without invented corroboration."""
    text = summarize(0.2, 0.9, 0.1, {})
    assert "SSL head synthetic score: 0.90" in text
    assert "Spectral head synthetic score: 0.10" in text
    assert "fusion weight" not in text
    assert "agrees" not in text
