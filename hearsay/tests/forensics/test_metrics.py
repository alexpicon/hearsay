# Author: Alex Picon <alexnpc@me.com>
"""Tests for AUC, EER and the two minDCF readings."""

import numpy as np
import pytest

from forensics.metrics import auc, eer, min_dcf


def _brute_dcf(y: np.ndarray, s: np.ndarray, w_miss: float, w_flag: float) -> float:
    """Compute a reference minDCF by trying every threshold explicitly."""
    best = 1.0
    for t in np.concatenate([[-np.inf], np.unique(s)]):
        flagged = s > t
        miss = np.mean(~flagged[y == 1])
        flag = np.mean(flagged[y == 0])
        best = min(best, w_miss * miss + w_flag * flag)
    return best


class TestMinDcf:
    """minDCF agrees with a brute-force reference."""

    def test_matches_brute_force(self) -> None:
        """Both readings match the explicit threshold sweep, with ties."""
        rng = np.random.default_rng(3)
        y = (rng.random(300) < 0.3).astype(int)
        s = np.round(rng.random(300) + 0.5 * y, 2)
        assert np.isclose(min_dcf(y, s, "nsa"), _brute_dcf(y, s, 1, 4))
        assert np.isclose(min_dcf(y, s, "code"), _brute_dcf(y, s, 4, 1))

    def test_perfect_and_constant(self) -> None:
        """Perfect separation costs 0; a constant score costs 1."""
        y = np.array([0, 0, 1, 1])
        assert min_dcf(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
        assert min_dcf(y, np.full(4, 0.5)) == 1.0

    @pytest.mark.parametrize("prior", [0.1, 0.3, 0.8])
    def test_nonbalanced_prior_uses_class_costs(self, prior: float) -> None:
        """The analyst reading puts cost four on real clips for every prior."""
        rng = np.random.default_rng(2)
        y = np.tile([0, 1], 60)
        scores = np.round(rng.random(len(y)) + 0.3 * y, 2)
        for reading, miss, flag in [
            ("nsa", prior, 4 * (1 - prior)),
            ("code", 4 * prior, 1 - prior),
        ]:
            reference = _brute_dcf(
                y, scores, miss / min(miss, flag), flag / min(miss, flag)
            )
            assert min_dcf(y, scores, reading, p_spoof=prior) == pytest.approx(
                reference
            )


@pytest.mark.parametrize("prior", [0.3, 0.5, 0.8])
def test_matches_detector_metric_costs(prior: float) -> None:
    """Apply identical priors and error meanings in both packages for unique scores."""
    from hearsay.metrics import evaluate_scores

    rng = np.random.default_rng(9)
    y = np.tile([0, 1], 50)
    scores = rng.normal(y * 0.5, 1)
    detector = evaluate_scores(scores, y, p_spoof=prior)
    assert min_dcf(y, scores, p_spoof=prior) == pytest.approx(detector["min_dcf"])
    assert min_dcf(y, scores, "code", p_spoof=prior) == pytest.approx(
        detector["min_dcf_as_coded"]
    )


class TestAucEer:
    """AUC and EER edge cases."""

    def test_auc_nan_for_single_class(self) -> None:
        """AUC is undefined with one class."""
        assert np.isnan(auc(np.zeros(10), np.arange(10)))

    def test_eer_perfect(self) -> None:
        """Separable scores give zero EER."""
        assert eer(np.array([0, 0, 1, 1]), np.array([0.0, 0.1, 0.9, 1.0])) == 0.0
