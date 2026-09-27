# Author: Alex Picon <alexnpc@me.com>
"""Audio decoding and small signal helpers shared by the experts."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

if TYPE_CHECKING:
    from forensics.finding import Context

log = logging.getLogger(__name__)

FRAME = 400
HOP = 160


def _ffmpeg_decode(path: Path, sr: int) -> np.ndarray:
    """Decode any container ffmpeg understands to mono float32.

    Args:
        path: Audio file.
        sr: Output sample rate.

    Returns:
        The decoded signal.

    """
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def load_audio(path: Path, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load a file as mono float32 at ``sr``.

    WAV/FLAC/OGG go through libsndfile; everything else (MP3, M4A, MP4) is
    decoded by ffmpeg.

    Args:
        path: Audio file.
        sr: Target sample rate.

    Returns:
        The signal and the file's native sample rate.

    """
    try:
        y, native = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except sf.LibsndfileError, RuntimeError:
        log.debug("libsndfile could not read %s, using ffmpeg", path)
        native = int(probe_sample_rate(path) or sr)
        return _ffmpeg_decode(path, sr), native
    if native != sr:
        g = np.gcd(native, sr)
        y = resample_poly(y, sr // g, native // g).astype(np.float32)
    return y, int(native)


def probe_sample_rate(path: Path) -> int | None:
    """Ask ffprobe for the first audio stream's sample rate.

    Args:
        path: Audio file.

    Returns:
        Sample rate in Hz, or None when ffprobe fails.

    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path),
    ]  # fmt: skip
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().split(",")[0])
    except ValueError, IndexError, subprocess.SubprocessError:
        return None


def frame_db(y: np.ndarray, frame: int = FRAME, hop: int = HOP) -> np.ndarray:
    """Return per-frame RMS level in dBFS.

    Args:
        y: Signal.
        frame: Frame length in samples.
        hop: Hop in samples.

    Returns:
        One dB value per frame.

    """
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    power = np.mean(y[idx].astype(np.float64) ** 2, axis=1)
    return 10 * np.log10(power + 1e-12)


def speech_mask(ctx: Context) -> np.ndarray:
    """Return a boolean speech/non-speech mask on the 10 ms frame grid.

    A frame is speech when it is within 25 dB of the loud frames and at least
    8 dB above the estimated noise floor. The mask is cached on the context.

    Args:
        ctx: File context.

    Returns:
        Boolean array, one value per 10 ms frame.

    """
    if "speech_mask" not in ctx.cache:
        db = levels(ctx)
        loud = np.percentile(db, 95)
        floor = np.percentile(db, 10)
        thr = max(loud - 25.0, floor + 8.0)
        mask = db > thr
        kernel = np.ones(5) / 5
        mask = np.convolve(mask.astype(float), kernel, mode="same") > 0.4
        ctx.cache["speech_mask"] = mask
    return ctx.cache["speech_mask"]


def levels(ctx: Context) -> np.ndarray:
    """Return cached per-frame levels (dBFS) for the context's audio.

    Args:
        ctx: File context.

    Returns:
        Per-frame dB values on the 10 ms grid.

    """
    if "levels" not in ctx.cache:
        ctx.cache["levels"] = frame_db(ctx.audio)
    return ctx.cache["levels"]


def power_spectrogram(
    ctx: Context,
    nperseg: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a cached power spectrogram with a 10 ms hop.

    Args:
        ctx: File context.
        nperseg: FFT window length.

    Returns:
        Frequencies (Hz) and power array shaped (freqs, frames).

    """
    key = f"spec{nperseg}"
    if key not in ctx.cache:
        f, _, z = stft(
            ctx.audio,
            fs=ctx.sr,
            nperseg=nperseg,
            noverlap=nperseg - HOP,
            boundary=None,
            padded=False,
        )
        ctx.cache[key] = (f, np.abs(z) ** 2 + 1e-14)
    return ctx.cache[key]


def aligned(mask: np.ndarray, n: int) -> np.ndarray:
    """Pad or cut a frame mask so it matches ``n`` frames.

    Args:
        mask: Boolean mask.
        n: Required length.

    Returns:
        Mask of length ``n``.

    """
    if len(mask) >= n:
        return mask[:n]
    return np.pad(mask, (0, n - len(mask)), constant_values=False)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs of consecutive True runs.

    Args:
        mask: Boolean array.

    Returns:
        List of half-open intervals.

    """
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))
