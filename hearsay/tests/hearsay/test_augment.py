# Author: Alex Picon <alexnpc@me.com>
"""Tests for the simulated test-set channel."""

import numpy as np
from scipy.signal import welch

from hearsay.augment import ChannelConfig, add_noise, lowpass, simulate_channel


def _tone_mix(n: int = 32000) -> np.ndarray:
    """Return a deterministic broadband test signal."""
    rng = np.random.default_rng(1)
    return (0.3 * rng.standard_normal(n)).astype(np.float32)


class TestChannel:
    """Shape, level and band-limiting of the simulated channel."""

    def test_shape_and_peak(self) -> None:
        """Output keeps the length and is peak-normalized near 0 dBFS."""
        x = _tone_mix()
        y = simulate_channel(x, np.random.default_rng(0))
        assert y.shape == x.shape
        assert 0.94 <= np.abs(y).max() <= 1.0

    def test_noise_hits_requested_snr(self) -> None:
        """Added noise power matches the requested SNR within 0.5 dB."""
        x = _tone_mix()
        y = add_noise(x, 20.0, np.random.default_rng(0))
        snr = 10 * np.log10(np.mean(x**2) / np.mean((y - x) ** 2))
        assert abs(snr - 20.0) < 0.5

    def test_lowpass_removes_top_band(self) -> None:
        """The low-pass leaves little energy above 7.6 kHz."""
        x = _tone_mix()
        y = lowpass(x, 7000.0, np.random.default_rng(0))
        f, p = welch(y, 16000, nperseg=1024)
        top = p[f > 7700].mean() / p[(f > 2000) & (f < 5000)].mean()
        assert top < 0.05

    def test_disabled_channel_only_normalizes(self) -> None:
        """With every step disabled the clip is only rescaled and quantized."""
        cfg = ChannelConfig(p_noise=0, p_lowpass=0, p_codec=0, p_resample=0)
        x = _tone_mix()
        y = simulate_channel(x, np.random.default_rng(0), cfg)
        corr = np.corrcoef(x, y)[0, 1]
        assert corr > 0.999


class TestRoomTone:
    """Room tone is low-frequency heavy and optional."""

    def test_room_tone_is_low_frequency(self) -> None:
        """Most of the added room-tone power sits below 300 Hz."""
        from hearsay.augment import add_room_tone

        x = _tone_mix()
        n = add_room_tone(x, 20.0, np.random.default_rng(0)) - x
        f, p = welch(n, 16000, nperseg=2048)
        assert p[f < 300].sum() / p.sum() > 0.8

    def test_default_config_keeps_random_stream(self) -> None:
        """With p_room = 0 the output equals the pre-room-tone behaviour."""
        x = _tone_mix()
        a = simulate_channel(x, np.random.default_rng(3), ChannelConfig(p_codec=0))
        b = simulate_channel(
            x, np.random.default_rng(3), ChannelConfig(p_codec=0, p_room=0.0)
        )
        np.testing.assert_array_equal(a, b)
