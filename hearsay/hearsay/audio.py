# Author: Alex Picon <alexnpc@me.com>
"""Audio loading helpers: any container in, 16 kHz mono float32 out."""

import io
import logging
import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
logger = logging.getLogger(__name__)


class AudioDecodeError(RuntimeError):
    """Raised when a file cannot be decoded by soundfile or ffmpeg."""


def resample(x: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Resample a mono signal with a polyphase filter.

    Args:
        x: Mono waveform.
        sr: Sample rate of ``x``.
        target: Output sample rate.

    Returns:
        The resampled waveform as float32.

    """
    if sr == target:
        return x.astype(np.float32, copy=False)
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def _ffmpeg_decode(data: bytes | None, path: Path | None) -> np.ndarray:
    """Decode arbitrary audio with ffmpeg into 16 kHz mono float32.

    Args:
        data: Raw file bytes (used when ``path`` is None).
        path: File path to decode.

    Returns:
        The decoded waveform.

    Raises:
        AudioDecodeError: If ffmpeg is missing or fails.

    """
    if shutil.which("ffmpeg") is None:
        msg = "ffmpeg is not installed and soundfile could not decode the audio"
        raise AudioDecodeError(msg)
    src = str(path) if path is not None else "pipe:0"
    cmd = ["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", str(SAMPLE_RATE)]
    cmd += ["-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, input=data, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        msg = f"ffmpeg failed: {proc.stderr.decode(errors='ignore')[:200]}"
        raise AudioDecodeError(msg)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(source: str | Path | bytes) -> np.ndarray:
    """Load audio from a path or raw bytes as 16 kHz mono float32.

    Args:
        source: A file path or the raw bytes of an audio file.

    Returns:
        The waveform in [-1, 1] at 16 kHz.

    Raises:
        AudioDecodeError: If the audio cannot be decoded or is empty.

    """
    is_bytes = isinstance(source, bytes)
    try:
        handle = io.BytesIO(source) if is_bytes else str(source)
        x, sr = sf.read(handle, dtype="float32", always_2d=True)
        x = resample(x.mean(axis=1), sr)
    except (sf.LibsndfileError, RuntimeError, ValueError) as err:
        logger.debug("soundfile failed (%s); falling back to ffmpeg", err)
        x = _ffmpeg_decode(
            source if is_bytes else None, None if is_bytes else Path(source)
        )
    if x.size == 0:
        msg = "decoded audio is empty"
        raise AudioDecodeError(msg)
    return np.nan_to_num(x).astype(np.float32, copy=False)


def peak_normalize(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale a waveform so its absolute peak equals ``peak``.

    Args:
        x: Waveform.
        peak: Target absolute peak.

    Returns:
        The scaled waveform (unchanged if silent).

    """
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x if m < 1e-9 else (x * (peak / m)).astype(np.float32)


def crop(x: np.ndarray, seconds: float, rng: np.random.Generator | None) -> np.ndarray:
    """Crop a waveform to at most ``seconds`` (random offset if rng is given).

    Args:
        x: Waveform.
        seconds: Maximum duration.
        rng: Random generator for the offset; None keeps the start.

    Returns:
        The cropped waveform.

    """
    n = int(seconds * SAMPLE_RATE)
    if x.size <= n:
        return x
    start = int(rng.integers(0, x.size - n + 1)) if rng is not None else 0
    return x[start : start + n]
