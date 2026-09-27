# Author: Alex Picon <alexnpc@me.com>
"""Spectral / frequency-domain expert.

Looks at the long-term spectrum and at frame statistics of the speech
regions: effective bandwidth and how sharply it ends (band-limiting), spectral
roll-off, flatness and tilt, high-band energy, stationary tonal lines that
neural vocoders leave behind, cepstral peak prominence (harmonic clarity) and
how the high band tracks the low band over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import welch

from forensics.audio import aligned, power_spectrogram, speech_mask
from forensics.expert import run_extractor
from forensics.finding import Context, Finding

DESCRIPTIONS = {
    "bandwidth_hz": "effective bandwidth",
    "cutoff_drop_db": "steepness of the band edge",
    "rolloff95_hz": "95% spectral roll-off",
    "flatness_speech": "spectral flatness in speech",
    "flatness_quiet": "spectral flatness in pauses",
    "hf_ratio_db": "4-7 kHz energy relative to 0.3-4 kHz",
    "ltas_slope_db_oct": "spectral tilt (dB/octave)",
    "flux": "frame-to-frame spectral change",
    "tonal_lines": "stationary tonal lines above 2 kHz",
    "cpp_db": "cepstral peak prominence (harmonic clarity)",
    "hf_lf_env_corr": "high/low band envelope correlation",
    "centroid_std_hz": "variability of the spectral centroid",
}


def _band(f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return a boolean frequency mask for [lo, hi).

    Args:
        f: Frequency axis.
        lo: Lower edge in Hz.
        hi: Upper edge in Hz.

    Returns:
        Boolean mask.

    """
    return (f >= lo) & (f < hi)


def _cpp(y: np.ndarray, sr: int, mask: np.ndarray) -> float:
    """Median cepstral peak prominence over speech frames.

    Args:
        y: Signal.
        sr: Sample rate.
        mask: Speech mask on the 10 ms grid.

    Returns:
        CPP in dB (NaN when there is too little speech).

    """
    n = 1024
    starts = np.flatnonzero(mask)[::3] * 160
    starts = starts[starts + n <= len(y)]
    if len(starts) < 5:
        return float("nan")
    frames = y[starts[:, None] + np.arange(n)[None, :]] * np.hanning(n)
    logmag = 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-9)
    ceps = np.fft.irfft(logmag, axis=1)[:, : n // 2]
    q = np.arange(n // 2) / sr
    sel = (q >= 1 / 400) & (q <= 1 / 60)
    peaks = ceps[:, sel].max(axis=1)
    fit = np.polyfit(q[sel], ceps[:, sel].T, 1)
    base = fit[0] * q[sel][ceps[:, sel].argmax(axis=1)] + fit[1]
    return float(np.median(peaks - base))


def is_band_limited(feats: dict[str, float], sr: int) -> bool:
    """Return True for a brick-wall band edge below the container's Nyquist.

    Args:
        feats: Spectral features (``bandwidth_hz`` and ``cutoff_drop_db``).
        sr: Sample rate of the decoded audio.

    Returns:
        Whether the clip was band-limited before it was saved.

    """
    bandwidth = feats.get("bandwidth_hz", sr / 2)
    return bool(bandwidth < 0.97 * sr / 2 and feats.get("cutoff_drop_db", 0) > 25)


def extract(ctx: Context) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure the spectral features of one clip.

    Args:
        ctx: File context.

    Returns:
        Features and report evidence (including the long-term spectrum).

    """
    f, p = power_spectrogram(ctx)
    mask = aligned(speech_mask(ctx), p.shape[1])
    if mask.sum() < 10:
        mask = np.ones(p.shape[1], dtype=bool)
    quiet = ~mask
    sp = p[:, mask]
    ltas = 10 * np.log10(sp.mean(axis=1))
    ref = ltas[_band(f, 300, 3000)].mean()
    above = np.flatnonzero(ltas - ref > -35)
    bandwidth = float(f[above.max()]) if len(above) else 0.0
    edge = _band(f, bandwidth + 150, ctx.sr / 2 - 40)
    drop = ltas[_band(f, 5500, 6800)].mean() - (
        ltas[edge].mean() if edge.any() else ltas[-3:].mean()
    )
    cum = np.cumsum(sp, axis=0) / sp.sum(axis=0, keepdims=True)
    rolloff = f[np.argmax(cum >= 0.95, axis=0)]
    band = _band(f, 300, min(7000, bandwidth) if bandwidth > 1000 else 7000)
    flat = np.exp(np.log(p[band]).mean(axis=0)) / p[band].mean(axis=0)
    hf = p[_band(f, 4000, 7000)].sum(axis=0)
    lf = p[_band(f, 300, 4000)].sum(axis=0)
    octaves = np.log2(f[_band(f, 500, 4000)] / 500)
    slope = np.polyfit(octaves, ltas[_band(f, 500, 4000)], 1)[0]
    logp = np.log(sp + 1e-12)
    flux = np.mean(np.abs(np.diff(logp, axis=1))) if sp.shape[1] > 2 else np.nan
    _, pw = welch(ctx.audio, fs=ctx.sr, nperseg=2048)
    ldb = 10 * np.log10(pw + 1e-20)
    resid = ldb - median_filter(ldb, size=31)
    fw = np.linspace(0, ctx.sr / 2, len(pw))
    hi = _band(fw, 2000, min(7200, max(bandwidth, 2500)))
    tonal = int(np.sum(resid[hi] > 8.0))
    centroid = (f[:, None] * sp).sum(axis=0) / sp.sum(axis=0)
    env_hf = 10 * np.log10(hf[mask] + 1e-12)
    env_lf = 10 * np.log10(p[_band(f, 300, 2000)][:, mask].sum(axis=0) + 1e-12)
    corr = np.corrcoef(env_hf, env_lf)[0, 1] if mask.sum() > 5 else np.nan
    feats = {
        "bandwidth_hz": bandwidth,
        "cutoff_drop_db": float(drop),
        "rolloff95_hz": float(np.median(rolloff)),
        "flatness_speech": float(np.median(flat[mask])),
        "flatness_quiet": float(np.median(flat[quiet])) if quiet.sum() > 5 else np.nan,
        "hf_ratio_db": float(10 * np.log10(hf[mask].sum() / lf[mask].sum())),
        "ltas_slope_db_oct": float(slope),
        "flux": float(flux),
        "tonal_lines": float(tonal),
        "cpp_db": _cpp(ctx.audio, ctx.sr, speech_mask(ctx)),
        "hf_lf_env_corr": float(corr),
        "centroid_std_hz": float(np.std(centroid)),
    }
    step = max(1, len(f) // 128)
    evidence = {
        "band_limited": is_band_limited(feats, ctx.sr),
        "series": {"ltas_f": f[::step].tolist(), "ltas_db": ltas[::step].tolist()},
    }
    if evidence["band_limited"]:
        evidence["note"] = (
            f"Audio is band-limited at {bandwidth:.0f} Hz ({drop:.0f} dB edge), "
            "so the top of the spectrum carries no voice evidence."
        )
    return feats, evidence


def _confidence(ctx: Context, feats: dict[str, float]) -> float:
    """Spectral confidence grows with the amount of speech analysed.

    Args:
        ctx: File context.
        feats: Measured features.

    Returns:
        Confidence in [0, 1].

    """
    speech_s = float(speech_mask(ctx).sum()) / 100
    wide = 1.0 if feats["bandwidth_hz"] > 6000 else 0.6
    return min(1.0, 0.3 + 0.25 * speech_s) * wide


def analyze(path: Path, ctx: Context) -> Finding:
    """Run the spectral expert.

    Args:
        path: Audio file (already referenced by ``ctx``).
        ctx: File context.

    Returns:
        The spectral finding.

    """
    return run_extractor(ctx, "spectral", extract, DESCRIPTIONS, _confidence)
