# Author: Alex Picon <alexnpc@me.com>
"""Tests for the minDCF wrapper around the organizers' scorer."""

import numpy as np
import pytest

from hearsay import metrics


class TestLocalCopyMatchesOfficial:
    """The local fallback must reproduce the official functions exactly."""

    @pytest.mark.skipif(
        metrics.OFFICIAL is None, reason="official scorer not extracted"
    )
    def test_same_det_curve_and_mindcf(self) -> None:
        """Random scores give identical DET curves and minDCF."""
        rng = np.random.default_rng(0)
        tar, non = rng.normal(1, 1, 300), rng.normal(0, 1, 200)
        ours = metrics.det_curve(tar, non)
        theirs = metrics.OFFICIAL.compute_det_curve(tar, non)
        for a, b in zip(ours, theirs, strict=True):
            np.testing.assert_allclose(a, b)
        official, _ = metrics.OFFICIAL.compute_mindcf(
            *theirs, metrics.P_SPOOF, metrics.C_MISS, metrics.C_FA
        )
        assert metrics.min_dcf(ours[0], ours[1]) == pytest.approx(official)


class TestEvaluateScores:
    """Direction handling and edge cases."""

    def test_perfect_detector(self) -> None:
        """A perfect ranking gives minDCF 0 under both readings."""
        y = np.array([0] * 70 + [1] * 30)
        p = np.where(y == 1, 0.9, 0.1)
        m = metrics.evaluate_scores(p, y)
        assert m["min_dcf"] == 0.0
        assert m["min_dcf_as_coded"] == 0.0
        assert m["auc"] == 1.0

    def test_inverted_detector_is_worst(self) -> None:
        """Scores with the wrong direction give minDCF 1 (trivial system)."""
        y = np.array([0] * 70 + [1] * 30)
        p = np.where(y == 1, 0.1, 0.9)
        m = metrics.evaluate_scores(p, y)
        assert m["min_dcf"] == pytest.approx(1.0)
        assert m["auc"] == 0.0

    def test_constant_scores(self) -> None:
        """Uninformative constant scores are no better than the trivial system."""
        y = np.array([0] * 70 + [1] * 30)
        m = metrics.evaluate_scores(np.full(100, 0.006), y)
        assert m["min_dcf"] == pytest.approx(1.0)

    @pytest.mark.parametrize("prior", [0.3, 0.5, 0.8])
    @pytest.mark.parametrize("official", [False, True])
    def test_class_priors_and_costs(
        self, prior: float, official: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both cost readings agree with independent threshold enumeration."""
        if official and metrics.OFFICIAL is None:
            pytest.skip("official scorer not extracted")
        if not official:
            monkeypatch.setattr(metrics, "OFFICIAL", None)
        y = np.array([0, 0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.6])
        analyst, as_coded = [], []
        for threshold in [-np.inf, *scores, np.inf]:
            miss = np.mean(scores[y == 1] < threshold)
            false_alarm = np.mean(scores[y == 0] >= threshold)
            analyst.append(prior * miss + 4 * (1 - prior) * false_alarm)
            as_coded.append((1 - prior) * false_alarm + 4 * prior * miss)
        result = metrics.evaluate_scores(scores, y, p_spoof=prior)
        assert result["min_dcf"] == pytest.approx(
            min(analyst) / min(prior, 4 * (1 - prior))
        )
        assert result["min_dcf_as_coded"] == pytest.approx(
            min(as_coded) / min(1 - prior, 4 * prior)
        )
        assert result["p_spoof"] == prior
        assert result["c_miss"] == 1
        assert result["c_fa"] == 4

    def test_logits_remain_supported(self) -> None:
        """Historic training callers can still evaluate unbounded logits."""
        result = metrics.evaluate_scores(np.array([-8.0, 12.0]), np.array([0, 1]))
        assert result["auc"] == 1.0
        assert result["p_spoof"] == 0.5

    @pytest.mark.parametrize(
        ("scores", "labels"),
        [
            ([], []),
            ([0.1, np.nan], [0, 1]),
            ([0.1, np.inf], [0, 1]),
            ([[0.1, 0.9]], [0, 1]),
            ([0.1, 0.9], [[0, 1]]),
            ([0.1, 0.9], [0]),
            ([0.1, 0.9], [0, 0]),
            ([0.1, 0.9], [1, 1]),
            ([0.1, 0.9], [0, 1.5]),
            ([0.1, 0.9], [0, -1]),
            ([0.1, 0.9], [0, np.nan]),
            ([0.1, 0.9], ["bonafide", "spoof"]),
            ([0.1, 0.9j], [0, 1]),
            ([0.1, 0.9], [0j, 1 + 0j]),
        ],
    )
    def test_invalid_input_rejected(self, scores: list, labels: list) -> None:
        """Malformed labels/scores cannot produce plausible evaluation metrics."""
        with pytest.raises(ValueError):
            metrics.evaluate_scores(np.asarray(scores), np.asarray(labels))

    @pytest.mark.parametrize("prior", [0, 1, -0.3, 1.2, np.nan, np.inf])
    def test_invalid_prior_rejected(self, prior: float) -> None:
        """Degenerate priors cannot divide by a zero or invalid default cost."""
        with pytest.raises(ValueError, match="p_spoof"):
            metrics.evaluate_scores(
                np.array([0.1, 0.9]), np.array([0, 1]), p_spoof=prior
            )


@pytest.mark.parametrize("target", [[], [np.nan], [[0.2]], [np.inf]])
def test_det_curve_requires_finite_classes(target: list) -> None:
    """The public DET helper rejects absent or invalid class scores."""
    with pytest.raises(ValueError):
        metrics.det_curve(np.asarray(target), np.array([0.1]))


@pytest.mark.parametrize(
    ("frr", "far"), [([], []), ([0, 1], [0]), ([0, np.nan], [1, 0]), ([-1], [0])]
)
def test_min_dcf_requires_valid_rates(frr: list, far: list) -> None:
    """Invalid rate vectors cannot silently yield an invalid DCF."""
    with pytest.raises(ValueError):
        metrics.min_dcf(np.asarray(frr), np.asarray(far))
