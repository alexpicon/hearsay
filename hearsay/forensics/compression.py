# Author: Alex Picon <alexnpc@me.com>
"""Compression forensics: traces of earlier lossy coding and transcoding.

Lossy codecs throw away quiet spectral detail (spectral holes), cut the top
of the spectrum with a steep low-pass, and change the statistics of the
samples. A file that says "PCM" but shows these traces was transcoded. The
idempotency test re-encodes the clip with MP3 and measures how much it
changes: audio that already went through a similar codec changes less.
Laundered fakes are often transcoded, but so are many real recordings, so
this expert is supporting evidence, not a verdict on its own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import correlate

from forensics.audio import aligned, power_spectrogram, speech_mask
from forensics.expert import run_extractor
from forensics.finding import Context, Finding

DESCRIPTIONS = {
    "spectral_holes": "share of spectral holes in speech",
    "hole_frames": "share of speech frames with many holes",
    "edge_db_per_khz": "slope of the band edge (dB/kHz)",
    "hist_empty_frac": "gaps in the sample-value histogram",
    "odd_sample_frac": "share of odd sample values",
    "peak_dbfs": "peak level (dBFS)",
    "clip_frac": "share of clipped samples",
    "recompress_snr_db": "change after MP3 re-encoding (SNR)",
}


def _mp3_roundtrip(y: np.ndarray, sr: int, kbps: int = 64) -> np.ndarray | None:
    """Encode and decode a signal with MP3 via ffmpeg.

    Args:
        y: Signal in [-1, 1].
        sr: Sample rate.
        kbps: Bit rate.

    Returns:
        The decoded signal (same length), or None if ffmpeg is unavailable.

    """
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    enc = [
        "ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1",
        "-i", "-", "-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3", "-",
    ]  # fmt: skip
    dec = [
        "ffmpeg", "-v", "error", "-i", "-", "-f", "s16le", "-ar", str(sr),
        "-ac", "1", "-",
    ]  # fmt: skip
    try:
        mp3 = subprocess.run(enc, input=pcm, capture_output=True, timeout=60).stdout
        raw = subprocess.run(dec, input=mp3, capture_output=True, timeout=60).stdout
    except OSError, subprocess.SubprocessError:
        return None
    out = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767
    if len(out) < len(y) // 2:
        return None
    seg = min(len(y), 2 * sr)
    corr = correlate(out[: seg + 2400], y[:seg], mode="valid", method="fft")
    lag = int(np.argmax(corr))
    out = out[lag : lag + len(y)]
    return np.pad(out, (0, len(y) - len(out)))


def _recompress_snr(y: np.ndarray, sr: int, cutoff: float) -> float:
    """SNR (dB) between the clip and its MP3 round trip, below the cutoff.

    The comparison is done on magnitude spectra so the codec delay does not
    matter.

    Args:
        y: Signal.
        sr: Sample rate.
        cutoff: Effective bandwidth in Hz.

    Returns:
        Spectral SNR in dB, or NaN.

    """
    rt = _mp3_roundtrip(y, sr)
    if rt is None:
        return float("nan")
    n = 512
    k = int(cutoff / (sr / 2) * (n // 2))

    def mag(x: np.ndarray) -> np.ndarray:
        frames = np.lib.stride_tricks.sliding_window_view(x, n)[::256] * np.hanning(n)
        return np.abs(np.fft.rfft(frames, axis=1))[:, 1:k]

    a, b = mag(y), mag(rt)
    m = min(len(a), len(b))
    err = np.sum((a[:m] - b[:m]) ** 2)
    return float(10 * np.log10(np.sum(a[:m] ** 2) / (err + 1e-12)))


def extract(ctx: Context) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure compression and transcoding traces.

    Args:
        ctx: File context (uses the spectral finding's bandwidth if present).

    Returns:
        Features and evidence.

    """
    y, sr = ctx.audio, ctx.sr
    f, p = power_spectrogram(ctx)
    mask = aligned(speech_mask(ctx), p.shape[1])
    spectral = ctx.findings.get("spectral")
    ltas = 10 * np.log10(p[:, mask].mean(axis=1) if mask.any() else p.mean(axis=1))
    if spectral is not None:
        cutoff = spectral.features.get("bandwidth_hz", sr / 2)
    else:
        ref = ltas[(f > 300) & (f < 3000)].mean()
        above = np.flatnonzero(ltas - ref > -35)
        cutoff = float(f[above.max()]) if len(above) else sr / 2
    db = 10 * np.log10(p[:, mask] if mask.any() else p)
    local = median_filter(db, size=(9, 1))
    band = (f > 1000) & (f < max(cutoff - 200, 1200))
    holes = (db[band] < local[band] - 30).astype(float)
    lo = (f > cutoff - 400) & (f < cutoff - 100)
    hi = (f > cutoff + 100) & (f < min(cutoff + 400, sr / 2))
    edge = (ltas[lo].mean() - ltas[hi].mean()) / 0.5 if hi.any() else np.nan
    q = np.round(y * 32768).astype(np.int32)
    small = q[np.abs(q) < 512]
    hist = np.bincount(small + 512, minlength=1024) if small.size else np.zeros(1)
    expected = hist > 0
    span = np.flatnonzero(expected)
    empty = np.nan
    if span.size > 10:
        inner = hist[span[0] : span[-1] + 1]
        empty = float(np.mean(inner == 0))
    feats = {
        "spectral_holes": float(holes.mean()) if holes.size else np.nan,
        "hole_frames": float(np.mean(holes.mean(axis=0) > 0.1))
        if holes.size
        else np.nan,
        "edge_db_per_khz": float(edge),
        "hist_empty_frac": empty,
        "odd_sample_frac": float(np.mean(q % 2 != 0)),
        "peak_dbfs": float(20 * np.log10(np.max(np.abs(y)) + 1e-9)),
        "clip_frac": float(np.mean(np.abs(y) > 0.999)),
        "recompress_snr_db": _recompress_snr(y, sr, cutoff),
        "cutoff_hz": float(cutoff),
    }
    container = ctx.findings.get("container")
    lossy = bool(container and container.features.get("lossy_codec"))
    transcoded = cutoff < 0.9 * sr / 2 and (edge or 0) > 40
    evidence: dict[str, Any] = {"lossy_container": lossy, "transcoded": transcoded}
    if transcoded and not lossy:
        evidence["note"] = (
            f"PCM file with a {edge:.0f} dB/kHz brick-wall edge at {cutoff:.0f} Hz: "
            "it was resampled or lossy-coded before being saved as WAV."
        )
    return feats, evidence


def _confidence(ctx: Context, feats: dict[str, float]) -> float:
    """Compression traces are generic, so confidence stays moderate.

    Args:
        ctx: File context.
        feats: Measured features.

    Returns:
        Confidence in [0, 1].

    """
    return float(min(0.7, 0.2 + ctx.duration / 10))


def analyze(path: Path, ctx: Context) -> Finding:
    """Run the compression expert.

    Args:
        path: Audio file (already referenced by ``ctx``).
        ctx: File context.

    Returns:
        The compression finding.

    """
    return run_extractor(ctx, "compression", extract, DESCRIPTIONS, _confidence)
