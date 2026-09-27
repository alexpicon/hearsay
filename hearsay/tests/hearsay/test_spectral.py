# Author: Alex Picon <alexnpc@me.com>
"""Tests for the hand-crafted feature extractor."""

import numpy as np

from hearsay.spectral import feature_names, spectral_features


class TestSpectralFeatures:
    """Vector layout and robustness."""

    def test_length_matches_names(self) -> None:
        """Every value has a name."""
        x = np.random.default_rng(0).standard_normal(48000).astype(np.float32) * 0.1
        assert spectral_features(x).shape == (len(feature_names()),)

    def test_short_and_silent_input(self) -> None:
        """Very short or silent clips still produce finite features."""
        for x in (np.zeros(100, np.float32), np.zeros(16000, np.float32)):
            v = spectral_features(x)
            assert np.isfinite(v).all()

    def test_duration_feature(self) -> None:
        """The last feature is the duration in seconds."""
        x = np.random.default_rng(0).standard_normal(32000).astype(np.float32)
        assert feature_names()[-1] == "duration_s"
        assert spectral_features(x)[-1] == 2.0
