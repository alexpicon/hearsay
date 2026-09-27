# Author: Alex Picon <alexnpc@me.com>
"""Prosody and phonetics expert.

Human speech has a lively, slightly irregular pitch contour, a speaking
rhythm with pauses, and audible breaths between phrases. Many generators
produce contours that are too smooth or too steady, pauses of uniform length,
and no breaths. This expert measures those properties with pYIN pitch
tracking, the speech/pause segmentation and a simple breath detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.signal import find_peaks

from forensics.audio import aligned, levels, power_spectrogram, runs, speech_mask
from forensics.expert import run_extractor
from forensics.finding import Context, Finding

DESCRIPTIONS = {
    "f0_std_st": "pitch variability (semitones)",
    "f0_range_st": "pitch range (semitones)",
    "f0_delta_st": "frame-to-frame pitch movement",
    "f0_curv_st": "pitch contour curvature",
    "f0_steady_frac": "share of perfectly steady pitch frames",
    "f0_perturb": "pitch perturbation (jitter-like)",
    "amp_perturb": "amplitude perturbation (shimmer-like)",
    "voiced_ratio": "voiced share of speech",
    "vprob_mean": "voicing clarity",
    "declination_st_s": "pitch declination (semitones/s)",
    "pause_rate": "pauses per second",
    "pause_mean_s": "mean pause length",
    "pause_cv": "variation of pause lengths",
    "breaths_per_min": "breaths per minute",
    "syllable_rate": "syllable-like nuclei per second",
    "energy_std_db": "loudness variability in speech",
}


def _pitch(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    """Run pYIN on the clip (cached).

    Args:
        ctx: File context.

    Returns:
        F0 in Hz (NaN when unvoiced) and voicing probability per 10 ms frame.

    """
    if "pitch" not in ctx.cache:
        f0, _, prob = librosa.pyin(
            ctx.audio,
            fmin=60,
            fmax=450,
            sr=ctx.sr,
            frame_length=1024,
            hop_length=160,
        )
        ctx.cache["pitch"] = (f0, prob)
    return ctx.cache["pitch"]


def _breaths(ctx: Context, mask: np.ndarray) -> int:
    """Count breath-like segments inside pauses.

    A breath is a 100-700 ms pause segment that is clearly above the noise
    floor but well below speech, with a noisy (flat) spectrum centred between
    0.5 and 3.5 kHz.

    Args:
        ctx: File context.
        mask: Speech mask.

    Returns:
        Number of breath candidates.

    """
    db = levels(ctx)
    f, p = power_spectrogram(ctx)
    floor, loud = np.percentile(db, 5), np.percentile(db[mask], 90)
    count = 0
    for start, end in runs(~mask):
        if not 10 <= end - start <= 70 or start == 0 or end >= len(mask):
            continue
        seg = db[start:end]
        if not (seg.max() > floor + 6 and seg.max() < loud - 12):
            continue
        spec = p[:, start : min(end, p.shape[1])].mean(axis=1)
        if spec.size == 0 or not np.all(np.isfinite(spec)):
            continue
        centroid = float((f * spec).sum() / spec.sum())
        if 500 < centroid < 3500:
            count += 1
    return count


def extract(ctx: Context) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure prosodic features of one clip.

    Args:
        ctx: File context.

    Returns:
        Features and evidence (pitch contour for the report).

    """
    f0, prob = _pitch(ctx)
    mask = aligned(speech_mask(ctx), len(f0))
    voiced = np.isfinite(f0) & mask
    speech_s = max(mask.sum() / 100, 1e-3)
    st = 12 * np.log2(f0 / np.nanmedian(f0[voiced])) if voiced.any() else f0
    both = voiced[1:] & voiced[:-1]
    d = np.abs(np.diff(st))[both]
    d2 = np.abs(np.diff(st, 2))[voiced[2:] & voiced[1:-1] & voiced[:-2]]
    period = 1 / f0
    per_d = np.abs(np.diff(period))[both] / np.nanmean(period[voiced])
    amp = 10 ** (aligned_levels(ctx, len(f0)) / 20)
    amp_d = np.abs(np.diff(amp))[both] / (amp[voiced].mean() + 1e-9)
    t = np.flatnonzero(voiced) / 100
    decl = np.polyfit(t, st[voiced], 1)[0] if voiced.sum() > 20 else np.nan
    speech_runs = runs(mask)
    pauses = [
        (b - a) / 100 for a, b in runs(~mask) if a > 0 and b < len(mask) and b - a >= 15
    ]
    db = aligned_levels(ctx, len(f0))
    env = np.convolve(db, np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(np.where(voiced, env, -120), prominence=3, distance=8)
    pm = np.array(pauses) if pauses else np.array([np.nan])
    feats = {
        "f0_median_hz": float(np.nanmedian(f0[voiced])) if voiced.any() else np.nan,
        "f0_std_st": float(np.nanstd(st[voiced])),
        "f0_range_st": _range(st[voiced]),
        "f0_delta_st": float(np.median(d)) if d.size else np.nan,
        "f0_curv_st": float(np.median(d2)) if d2.size else np.nan,
        "f0_steady_frac": float(np.mean(d < 0.05)) if d.size else np.nan,
        "f0_perturb": float(np.median(per_d)) if per_d.size else np.nan,
        "amp_perturb": float(np.median(amp_d)) if amp_d.size else np.nan,
        "voiced_ratio": float(voiced.sum() / max(mask.sum(), 1)),
        "vprob_mean": float(np.nanmean(prob[voiced])) if voiced.any() else np.nan,
        "declination_st_s": float(decl),
        "pause_rate": len(pauses) / speech_s,
        "pause_mean_s": float(np.nanmean(pm)),
        "pause_cv": float(np.nanstd(pm) / np.nanmean(pm))
        if len(pauses) > 1
        else np.nan,
        "breaths_per_min": 60
        * _breaths(ctx, aligned(speech_mask(ctx), len(levels(ctx))))
        / max(len(ctx.audio) / ctx.sr, 1),
        "syllable_rate": len(peaks) / speech_s,
        "energy_std_db": float(np.std(db[mask])) if mask.any() else np.nan,
    }
    evidence = {
        "voiced_s": float(voiced.sum() / 100),
        "speech_segments": len(speech_runs),
        "series": {"f0": np.where(np.isfinite(f0), f0, 0).round(1).tolist()},
    }
    return feats, evidence


def _range(values: np.ndarray) -> float:
    """Return the 5-95 percentile range, or NaN with too few values.

    Args:
        values: Pitch values in semitones.

    Returns:
        Range in semitones.

    """
    values = values[np.isfinite(values)]
    if values.size < 5:
        return float("nan")
    hi, lo = np.percentile(values, [95, 5])
    return float(hi - lo)


def aligned_levels(ctx: Context, n: int) -> np.ndarray:
    """Return frame levels padded/cut to ``n`` frames.

    Args:
        ctx: File context.
        n: Number of frames required.

    Returns:
        dB levels of length ``n``.

    """
    db = levels(ctx)
    if len(db) >= n:
        return db[:n]
    return np.pad(db, (0, n - len(db)), constant_values=db.min())


def _confidence(ctx: Context, feats: dict[str, float]) -> float:
    """Prosody needs a few seconds of voiced speech to be meaningful.

    Args:
        ctx: File context.
        feats: Measured features.

    Returns:
        Confidence in [0, 1].

    """
    voiced_s = feats["voiced_ratio"] * speech_mask(ctx).sum() / 100
    return float(min(1.0, voiced_s / 3.0))


def analyze(path: Path, ctx: Context) -> Finding:
    """Run the prosody expert.

    Args:
        path: Audio file (already referenced by ``ctx``).
        ctx: File context.

    Returns:
        The prosody finding.

    """
    return run_extractor(ctx, "prosody", extract, DESCRIPTIONS, _confidence)
