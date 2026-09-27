# Author: Alex Picon <alexnpc@me.com>
"""Speaker-embedding consistency: checks whether the voice drifts in a clip.

The clip is cut into overlapping 1 s windows. Each window is embedded with
a speaker-verification network (WavLM-base-plus x-vectors) and the embeddings
are compared with each other. One real person stays one voice; voice
conversion and spliced or partially generated audio can drift or jump. When
the network is unavailable the expert reports MFCC measurements without a
calibrated score. The WavLM calibration does not apply to MFCC statistics.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from forensics.audio import aligned, speech_mask
from forensics.expert import run_extractor
from forensics.finding import Context, Finding

log = logging.getLogger(__name__)
MODEL_ID = os.environ.get("HEARSAY_SPEAKER_MODEL", "microsoft/wavlm-base-plus-sv")
WIN_S, HOP_S, MAX_WINDOWS = 1.0, 0.5, 8
DESCRIPTIONS = {
    "sim_mean": "mean similarity of windows to the whole-clip voice",
    "sim_min": "least similar window",
    "adjacent_min": "largest jump between neighbouring windows",
    "sim_std": "spread of window similarities",
    "drift_per_s": "voice drift per second",
}


@functools.cache
def _wavlm() -> tuple[Any, Any] | None:
    """Load the speaker-verification model once per process.

    Returns:
        (feature extractor, model), or None when the model cannot be loaded.

    """
    if os.environ.get("HEARSAY_SPEAKER_BACKEND", "wavlm") != "wavlm":
        return None
    try:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        torch.set_num_threads(int(os.environ.get("HEARSAY_TORCH_THREADS", "1")))
        fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        model = WavLMForXVector.from_pretrained(MODEL_ID).eval()
    except (ImportError, OSError, ValueError) as err:
        log.warning("speaker model unavailable (%s); using MFCC fallback", err)
        return None
    return fe, model


def _embed(windows: list[np.ndarray], sr: int) -> tuple[np.ndarray, str]:
    """Embed each window.

    Args:
        windows: Equal-length audio windows.
        sr: Sample rate.

    Returns:
        Unit-norm embeddings (n, d) and the backend name.

    """
    loaded = _wavlm()
    if loaded is not None:
        import torch

        fe, model = loaded
        with torch.inference_mode():
            inp = fe(windows, sampling_rate=sr, return_tensors="pt", padding=True)
            emb = model(**inp).embeddings.numpy()
        backend = "wavlm-xvector"
    else:
        emb = np.stack([_mfcc_stats(w, sr) for w in windows])
        backend = "mfcc-stats"
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return emb, backend


def _mfcc_stats(w: np.ndarray, sr: int) -> np.ndarray:
    """Return mean and spread of MFCCs 1-20 (fallback embedding, no energy).

    Args:
        w: Audio window.
        sr: Sample rate.

    Returns:
        40-dimensional vector.

    """
    m = librosa.feature.mfcc(y=w, sr=sr, n_mfcc=21)[1:]
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def extract(ctx: Context) -> tuple[dict[str, float], dict[str, Any]]:
    """Embed sliding windows and measure how consistent the voice is.

    Args:
        ctx: File context.

    Returns:
        Features and evidence (similarity curve for the report).

    """
    y, sr = ctx.audio, ctx.sr
    win, hop = int(WIN_S * sr), int(HOP_S * sr)
    mask = speech_mask(ctx)
    starts = []
    for s in range(0, max(1, len(y) - win + 1), hop):
        m = aligned(mask, len(y) // 160 + 1)[s // 160 : (s + win) // 160]
        if m.mean() >= 0.4:
            starts.append(s)
    nan = float("nan")
    if len(starts) < 3:
        feats = dict.fromkeys(DESCRIPTIONS, nan)
        return feats, {"windows": len(starts), "note": "Too little speech to compare."}
    if len(starts) > MAX_WINDOWS:
        pick = np.linspace(0, len(starts) - 1, MAX_WINDOWS).round().astype(int)
        starts = [starts[i] for i in pick]
    windows = [y[s : s + win] for s in starts]
    emb, backend = _embed(windows, sr)
    centroid = emb.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    to_centroid = emb @ centroid
    adjacent = np.sum(emb[1:] * emb[:-1], axis=1)
    to_first = emb @ emb[0]
    t = np.array(starts) / sr
    drift = float(-np.polyfit(t, to_first, 1)[0]) if len(t) > 2 else nan
    feats = {
        "sim_mean": float(to_centroid.mean()),
        "sim_min": float(to_centroid.min()),
        "adjacent_min": float(adjacent.min()),
        "sim_std": float(to_centroid.std()),
        "drift_per_s": drift,
    }
    evidence = {
        "backend": backend,
        "windows": len(starts),
        "series": {"t": t.round(2).tolist(), "sim": to_centroid.round(4).tolist()},
    }
    worst = int(np.argmin(adjacent))
    if adjacent[worst] < np.median(adjacent) - 0.15:
        evidence["note"] = f"Voice changes abruptly near {t[worst + 1]:.1f} s."
    return feats, evidence


def _confidence(ctx: Context, feats: dict[str, float]) -> float:
    """More windows give a more trustworthy consistency estimate.

    Args:
        ctx: File context.
        feats: Measured features.

    Returns:
        Confidence in [0, 1].

    """
    if not np.isfinite(feats.get("sim_mean", np.nan)):
        return 0.0
    return float(min(1.0, (ctx.duration - WIN_S) / 4))


def analyze(path: Path, ctx: Context) -> Finding:
    """Run the speaker-consistency expert.

    Args:
        path: Audio file (already referenced by ``ctx``).
        ctx: File context.

    Returns:
        The speaker finding.

    """
    return run_extractor(ctx, "speaker", extract, DESCRIPTIONS, _confidence)
