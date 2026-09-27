# Author: Alex Picon <alexnpc@me.com>
"""Splice / discontinuity expert.

Inserting a generated word into a real recording, or stitching generated
phrases together, leaves seams: isolated sample-level clicks, frames where
the phase of every frequency jumps at once, sudden DC-offset steps, level
jumps in the middle of a word, and pauses whose background noise does not
match the pause before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from forensics.audio import HOP, aligned, levels, power_spectrogram, runs, speech_mask
from forensics.expert import run_extractor
from forensics.finding import Context, Finding

DESCRIPTIONS = {
    "click_rate": "isolated sample-level clicks per second",
    "phase_break_rate": "broadband phase breaks per second",
    "phase_break_max": "strongest phase break (robust z)",
    "dc_jump_max": "largest DC-offset step (x noise RMS)",
    "level_jump_db": "largest level jump inside speech",
    "seam_max_db": "largest background change between pauses",
    "seam_mean_db": "mean background change between pauses",
}


def _clicks(y: np.ndarray, sr: int) -> float:
    """Count isolated second-difference spikes per second.

    Args:
        y: Signal.
        sr: Sample rate.

    Returns:
        Clicks per second.

    """
    d2 = np.abs(np.diff(y, 2))
    local = median_filter(d2, size=81) + 1e-6
    spikes = np.flatnonzero(d2 > 15 * local)
    if spikes.size == 0:
        return 0.0
    groups = 1 + int(np.sum(np.diff(spikes) > sr // 100))
    return groups / (len(y) / sr)


def _phase_breaks(ctx: Context) -> tuple[float, float, list[float]]:
    """Find frames where phase deviates from its expected advance everywhere.

    Args:
        ctx: File context.

    Returns:
        (breaks per second, strongest break z-score, break times in seconds).

    """
    from scipy.signal import stft

    n = 512
    f, _, z = stft(ctx.audio, fs=ctx.sr, nperseg=n, noverlap=n - HOP)
    mag = np.abs(z)
    phase = np.angle(z)
    k = np.arange(len(f))[:, None]
    expected = 2 * np.pi * k * HOP / n
    dev = np.angle(np.exp(1j * (np.diff(phase, axis=1) - expected)))
    strong = mag[:, 1:] > np.percentile(mag, 75)
    score = np.where(strong, np.abs(dev), np.nan)
    with np.errstate(all="ignore"):
        counts = np.isfinite(score).sum(axis=0)
        per_frame = np.where(
            counts > 0, np.nansum(score, axis=0) / np.maximum(counts, 1), np.nan
        )
    per_frame = np.nan_to_num(per_frame, nan=np.nanmedian(per_frame))
    med = np.median(per_frame)
    mad = np.median(np.abs(per_frame - med)) * 1.4826 + 1e-9
    zf = (per_frame - med) / mad
    hits = np.flatnonzero(zf > 5)
    times = (hits * HOP / ctx.sr).round(2).tolist()
    return len(hits) / ctx.duration, float(zf.max()) if zf.size else 0.0, times


def _seams(ctx: Context, mask: np.ndarray) -> tuple[float, float]:
    """Compare the background spectrum of consecutive pauses.

    Args:
        ctx: File context.
        mask: Speech mask on the spectrogram grid.

    Returns:
        (max, mean) mean-absolute band-level difference in dB.

    """
    f, p = power_spectrogram(ctx)
    edges = np.geomspace(100, ctx.sr / 2 - 100, 13)
    bands = [(f >= a) & (f < b) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    profiles = []
    for a, b in runs(~mask):
        if b - a >= 5:
            seg = p[:, a:b]
            profiles.append([10 * np.log10(seg[m].mean()) for m in bands])
    if len(profiles) < 2:
        return float("nan"), float("nan")
    prof = np.array(profiles)
    diffs = np.abs(np.diff(prof, axis=0)).mean(axis=1)
    return float(diffs.max()), float(diffs.mean())


def extract(ctx: Context) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure discontinuity features.

    Args:
        ctx: File context.

    Returns:
        Features and evidence (times of the strongest breaks).

    """
    y, sr = ctx.audio, ctx.sr
    rate, zmax, times = _phase_breaks(ctx)
    block = sr // 50
    nb = len(y) // block
    blocks = y[: nb * block].reshape(nb, block)
    dc = blocks.mean(axis=1)
    rms = blocks.std(axis=1)
    noise = np.percentile(rms, 10) + 1e-6
    db = levels(ctx)
    mask = aligned(speech_mask(ctx), len(db))
    inside = mask[1:] & mask[:-1]
    jumps = np.abs(np.diff(db))[inside]
    _, p = power_spectrogram(ctx)
    seam_max, seam_mean = _seams(ctx, aligned(speech_mask(ctx), p.shape[1]))
    feats = {
        "click_rate": _clicks(y, sr),
        "phase_break_rate": rate,
        "phase_break_max": zmax,
        "dc_jump_max": float(np.abs(np.diff(dc)).max() / noise) if nb > 2 else np.nan,
        "level_jump_db": float(jumps.max()) if jumps.size else np.nan,
        "seam_max_db": seam_max,
        "seam_mean_db": seam_mean,
    }
    evidence: dict[str, Any] = {"phase_break_times_s": times[:20]}
    if times:
        shown = ", ".join(f"{t:.2f}" for t in times[:5])
        evidence["note"] = f"Broadband phase breaks at {shown} s."
    return feats, evidence


def _confidence(ctx: Context, feats: dict[str, float]) -> float:
    """Splice evidence scales with clip length.

    Args:
        ctx: File context.
        feats: Measured features.

    Returns:
        Confidence in [0, 1].

    """
    return float(min(0.8, 0.2 + ctx.duration / 8))


def analyze(path: Path, ctx: Context) -> Finding:
    """Run the splice expert.

    Args:
        path: Audio file (already referenced by ``ctx``).
        ctx: File context.

    Returns:
        The splice finding.

    """
    return run_extractor(ctx, "splice", extract, DESCRIPTIONS, _confidence)
