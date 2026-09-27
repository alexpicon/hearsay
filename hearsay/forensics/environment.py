# Author: Alex Picon <alexnpc@me.com>
"""Acoustic-environment expert: mains hum (ENF) and background consistency.

Hum, noise-floor changes and near-zero segments describe the recording and
processing chain. They may occur in either genuine or synthetic speech and
are not proof of origin. Learned scores depend on the calibration corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import stft, welch

from forensics.audio import aligned, levels, power_spectrogram, runs, speech_mask
from forensics.expert import run_extractor
from forensics.finding import Context, Finding

DESCRIPTIONS = {
    "enf_prom_db": "mains-hum (50/60 Hz) prominence",
    "enf_std_hz": "mains-hum frequency wander",
    "floor_db": "background noise floor",
    "snr_db": "speech-to-background ratio",
    "floor_drift_db": "noise-floor change between halves",
    "floor_var_db": "noise-floor variability",
    "noise_colour_corr": "background spectrum similarity between halves",
    "digital_silence_frac": "share of near-zero audio",
    "offset_decay_db": "level drop 100 ms after speech ends",
}


def _hum_prominence(y: np.ndarray, sr: int) -> tuple[float, int]:
    """Return the strongest mains-hum family and its prominence.

    Args:
        y: Signal.
        sr: Sample rate.

    Returns:
        (mean prominence in dB over harmonics 1-3, nominal frequency).

    """
    nper = int(min(len(y), 4 * sr))
    f, p = welch(y, fs=sr, nperseg=nper)
    db = 10 * np.log10(p + 1e-20)
    best = (-np.inf, 0)
    for nominal in (50, 60):
        proms = []
        for h in (1, 2, 3):
            fc = nominal * h
            peak = db[(f > fc - 1) & (f < fc + 1)]
            side = db[((f > fc - 8) & (f < fc - 3)) | ((f > fc + 3) & (f < fc + 8))]
            if peak.size and side.size:
                proms.append(peak.max() - np.median(side))
        if proms and np.mean(proms) > best[0]:
            best = (float(np.mean(proms)), nominal)
    return best


def _enf_wander(y: np.ndarray, sr: int, nominal: int) -> float:
    """Track the hum frequency in 2 s windows and return its standard deviation.

    Args:
        y: Signal.
        sr: Sample rate.
        nominal: 50 or 60.

    Returns:
        Standard deviation of the tracked frequency in Hz.

    """
    nper = 2 * sr
    f, _, z = stft(y, fs=sr, nperseg=nper, noverlap=nper - sr // 2, nfft=8 * nper)
    band = (f > nominal - 1) & (f < nominal + 1)
    mag = np.abs(z[band])
    idx = mag.argmax(axis=0)
    track = f[band][idx]
    return float(np.std(track)) if len(track) >= 3 else float("nan")


def extract(ctx: Context) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure hum and background-consistency features.

    Args:
        ctx: File context.

    Returns:
        Features and evidence.

    """
    y, sr = ctx.audio, ctx.sr
    db = levels(ctx)
    mask = aligned(speech_mask(ctx), len(db))
    prom, nominal = _hum_prominence(y, sr)
    long_enough = len(y) >= 4 * sr
    wander = _enf_wander(y, sr, nominal) if long_enough and prom > 6 else np.nan
    floor = float(np.percentile(db, 10))
    half = len(db) // 2
    first, second = db[:half], db[half:]
    blocks = [db[i : i + 50].min() for i in range(0, len(db) - 25, 50)]
    f, p = power_spectrogram(ctx)
    q = ~aligned(mask, p.shape[1])
    qa, qb = q.copy(), q.copy()
    qa[p.shape[1] // 2 :] = False
    qb[: p.shape[1] // 2] = False
    corr = np.nan
    if qa.sum() > 5 and qb.sum() > 5:
        sa = 10 * np.log10(p[:, qa].mean(axis=1))
        sb = 10 * np.log10(p[:, qb].mean(axis=1))
        corr = float(np.corrcoef(sa, sb)[0, 1])
    zero = np.abs(y) < 2e-5
    zero_runs = [(a, b) for a, b in runs(zero) if b - a >= 40]
    silent = sum(b - a for a, b in zero_runs) / len(y)
    margin = sr // 20
    inside = [(a, b) for a, b in zero_runs if a > margin and b < len(y) - margin]
    decays = [
        db[e - 1] - db[min(e + 10, len(db) - 1)]
        for _, e in runs(mask)
        if e + 10 < len(db)
    ]
    feats = {
        "enf_prom_db": prom,
        "enf_std_hz": wander,
        "floor_db": floor,
        "snr_db": float(np.percentile(db, 95) - floor),
        "floor_drift_db": float(
            abs(np.percentile(first, 10) - np.percentile(second, 10))
        ),
        "floor_var_db": float(np.std(blocks)) if len(blocks) > 2 else np.nan,
        "noise_colour_corr": corr,
        "digital_silence_frac": float(silent),
        "offset_decay_db": float(np.median(decays)) if decays else np.nan,
    }
    evidence: dict[str, Any] = {
        "hum_nominal_hz": nominal,
        "internal_near_zero_runs": len(inside),
        "near_zero_threshold": 2e-5,
    }
    if inside:
        first = inside[0][0] / sr
        evidence["note"] = (
            f"{len(inside)} near-zero segment(s) inside the clip "
            f"(first at {first:.2f} s). Quiet recording, gating or editing can "
            "produce this; it does not establish synthetic speech."
        )
    if prom > 6:
        hum = f"{nominal} Hz mains hum present ({prom:.1f} dB)."
        evidence["note"] = f"{evidence.get('note', '')} {hum}".strip()
        if np.isfinite(wander):
            kind = "steady (digital?)" if wander < 0.005 else "wandering (grid-like)"
            evidence["note"] += f" Tracked frequency is {kind}, std {wander:.3f} Hz."
    evidence["series"] = {"level_db": db[::2].round(1).tolist()}
    return feats, evidence


def _confidence(ctx: Context, feats: dict[str, float]) -> float:
    """Environment evidence is weak on short clips with few pauses.

    Args:
        ctx: File context.
        feats: Measured features.

    Returns:
        Confidence in [0, 1].

    """
    pause_s = float((~speech_mask(ctx)).sum()) / 100
    return float(min(1.0, 0.2 + pause_s / 2.0) * min(1.0, ctx.duration / 4))


def analyze(path: Path, ctx: Context) -> Finding:
    """Run the acoustic-environment expert.

    Args:
        path: Audio file (already referenced by ``ctx``).
        ctx: File context.

    Returns:
        The environment finding.

    """
    return run_extractor(ctx, "environment", extract, DESCRIPTIONS, _confidence)
