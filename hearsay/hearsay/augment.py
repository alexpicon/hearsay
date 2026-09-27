# Author: Alex Picon <alexnpc@me.com>
"""Channel simulation that reproduces the HEARSAY test-set conditions.

Profiling the 1671 test clips showed three processing steps applied to every
file: additive colored noise (median global SNR about 24 dB, 5th percentile
about 17 dB, with a strong low-frequency rumble), a gradual low-pass whose
-10 dB point sits near 7.3-7.4 kHz, and peak normalization to about 0 dBFS.
Training clips go through a randomized version of the same chain so that the
detector cannot rely on cues the test channel destroys (for example the
7.5-8 kHz band) and learns cues that survive it.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, firwin

from hearsay.audio import SAMPLE_RATE, peak_normalize, resample

logger = logging.getLogger(__name__)
HAS_FFMPEG = shutil.which("ffmpeg") is not None


@dataclass(frozen=True)
class ChannelConfig:
    """Probabilities and ranges of the simulated channel.

    Attributes:
        p_noise: Probability of adding noise.
        snr_db: Range of the global signal-to-noise ratio.
        p_lowpass: Probability of the gradual low-pass filter.
        cutoff_hz: Range of the low-pass cutoff frequency.
        p_codec: Probability of an MP3 round trip through ffmpeg.
        p_resample: Probability of a resampling round trip.
        p_room: Share of the added noise that is recording-room tone (steep
            low-frequency rumble over a faint white floor, the noise VCTK's
            own recordings carry) instead of generic colored noise.
        room_snr_db: Range of the global SNR for room tone.

    """

    p_noise: float = 0.9
    snr_db: tuple[float, float] = (10.0, 35.0)
    p_lowpass: float = 0.85
    cutoff_hz: tuple[float, float] = (6900.0, 7600.0)
    p_codec: float = 0.25
    p_resample: float = 0.2
    p_room: float = 0.0
    room_snr_db: tuple[float, float] = (18.0, 32.0)


def colored_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f^beta noise with unit variance.

    Args:
        n: Number of samples.
        beta: Spectral exponent (0 white, 1 pink, 2 brown).
        rng: Random generator.

    Returns:
        The noise signal.

    """
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    f[0] = f[1] if n > 1 else 1.0
    spec *= f ** (-beta / 2.0)
    y = np.fft.irfft(spec, n)
    return y / (y.std() + 1e-12)


def rumble(n: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency rumble plus faint mains hum, unit variance.

    Args:
        n: Number of samples.
        rng: Random generator.

    Returns:
        The rumble signal.

    """
    taps = firwin(801, float(rng.uniform(80, 250)), fs=SAMPLE_RATE)
    y = fftconvolve(rng.standard_normal(n + 800), taps, mode="valid")[:n]
    t = np.arange(n) / SAMPLE_RATE
    mains = 50.0 if rng.random() < 0.5 else 60.0
    hum = sum(
        rng.uniform(0, 0.3) / k * np.sin(2 * np.pi * mains * k * t + rng.uniform(0, 6))
        for k in (1, 2, 3)
    )
    y = y / (y.std() + 1e-12) + hum
    return y / (y.std() + 1e-12)


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add a random mix of colored noise and rumble at a global SNR.

    Args:
        x: Clean waveform.
        snr_db: Target signal-to-noise ratio in dB.
        rng: Random generator.

    Returns:
        The noisy waveform.

    """
    noise = colored_noise(x.size, float(rng.uniform(0.0, 2.0)), rng)
    if rng.random() < 0.7:
        noise = noise + float(rng.uniform(0.5, 3.0)) * rumble(x.size, rng)
    p_sig = float(np.mean(x.astype(np.float64) ** 2)) + 1e-12
    p_noise = float(np.mean(noise**2)) + 1e-12
    scale = np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10.0)))
    return (x + scale * noise).astype(np.float32)


def add_room_tone(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add recording-room tone: brown-noise rumble over a faint white floor.

    Args:
        x: Clean waveform.
        snr_db: Target global signal-to-noise ratio in dB.
        rng: Random generator.

    Returns:
        The noisy waveform.

    """
    tone = colored_noise(x.size, float(rng.uniform(1.7, 2.3)), rng)
    tone = tone + float(rng.uniform(0.005, 0.04)) * colored_noise(x.size, 0.0, rng)
    p_sig = float(np.mean(x.astype(np.float64) ** 2)) + 1e-12
    scale = np.sqrt(p_sig / ((float(np.mean(tone**2)) + 1e-12) * 10 ** (snr_db / 10)))
    return (x + scale * tone).astype(np.float32)


def lowpass(x: np.ndarray, cutoff: float, rng: np.random.Generator) -> np.ndarray:
    """Apply a short Hann-windowed sinc low-pass (gradual transition).

    Args:
        x: Waveform.
        cutoff: Cutoff frequency in Hz.
        rng: Random generator for the filter length.

    Returns:
        The filtered waveform.

    """
    numtaps = int(rng.choice([33, 49, 65, 97, 129]))
    taps = firwin(numtaps, cutoff, window="hann", fs=SAMPLE_RATE)
    return fftconvolve(x, taps, mode="same").astype(np.float32)


def mp3_roundtrip(x: np.ndarray, bitrate: str) -> np.ndarray:
    """Encode to MP3 and decode back with ffmpeg (no-op without ffmpeg).

    Args:
        x: Waveform at 16 kHz.
        bitrate: LAME bitrate such as ``"48k"``.

    Returns:
        The decoded waveform, trimmed or padded to the input length.

    """
    if not HAS_FFMPEG:
        return x
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    raw = ["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1"]
    enc = ["ffmpeg", "-v", "error", *raw, "-i", "pipe:0", "-c:a", "libmp3lame"]
    enc += ["-b:a", bitrate, "-f", "mp3", "pipe:1"]
    mp3 = subprocess.run(enc, input=pcm, capture_output=True, check=False).stdout
    dec = ["ffmpeg", "-v", "error", "-i", "pipe:0", *raw, "pipe:1"]
    out = subprocess.run(dec, input=mp3, capture_output=True, check=False).stdout
    y = np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0
    if y.size < x.size // 2:
        logger.debug("mp3 round trip failed; keeping the input")
        return x
    return np.pad(y, (0, max(0, x.size - y.size)))[: x.size]


def simulate_channel(
    x: np.ndarray,
    rng: np.random.Generator,
    cfg: ChannelConfig | None = None,
) -> np.ndarray:
    """Pass a clean clip through a randomized test-like channel.

    Args:
        x: Waveform at 16 kHz.
        rng: Random generator.
        cfg: Channel configuration (defaults to ``ChannelConfig()``).

    Returns:
        The degraded waveform, peak-normalized and quantized to 16 bits.

    """
    cfg = cfg or ChannelConfig()
    y = peak_normalize(x, 0.99)
    if rng.random() < cfg.p_noise:
        # p_room is checked first so configs without room tone keep the
        # exact random stream (and features) of earlier runs.
        if cfg.p_room > 0 and rng.random() < cfg.p_room:
            y = add_room_tone(y, float(rng.uniform(*cfg.room_snr_db)), rng)
        else:
            y = add_noise(y, float(rng.uniform(*cfg.snr_db)), rng)
    if rng.random() < cfg.p_resample:
        mid = int(rng.choice([8000 * 3 // 2, 22050, 24000, 44100]))
        y = resample(resample(y, SAMPLE_RATE, mid), mid, SAMPLE_RATE)[: x.size]
    if rng.random() < cfg.p_codec:
        y = mp3_roundtrip(
            peak_normalize(y, 0.98), str(rng.choice(["32k", "48k", "64k"]))
        )
    if rng.random() < cfg.p_lowpass:
        y = lowpass(y, float(rng.uniform(*cfg.cutoff_hz)), rng)
    y = np.clip(peak_normalize(y, float(rng.uniform(0.95, 1.0))), -1.0, 32767 / 32768)
    return (np.round(y * 32768) / 32768).astype(np.float32)
