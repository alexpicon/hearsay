# Author: Alex Picon <alexnpc@me.com>
"""Explainable hand-crafted acoustic features.

Each feature has a human-readable name so the gradient-boosted model's
per-clip SHAP contributions can be reported as evidence (for example
"high-band flatness is unusually low", a typical neural-vocoder trace).
"""

import numpy as np
from scipy.fft import dct
from scipy.signal import stft

from hearsay.audio import SAMPLE_RATE

N_FFT = 512
HOP = 160
N_LFCC = 20
BANDS = [
    (0, 150),
    (150, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000),
    (5000, 6000),
    (6000, 7000),
    (7000, 8000),
]


def _linear_filterbank(n_filters: int = 40) -> np.ndarray:
    """Triangular filters equally spaced on a linear frequency axis."""
    n_bins = N_FFT // 2 + 1
    edges = np.linspace(0, n_bins - 1, n_filters + 2)
    fb = np.zeros((n_filters, n_bins))
    bins = np.arange(n_bins)
    for i in range(n_filters):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (bins - lo) / (mid - lo)
        down = (hi - bins) / (hi - mid)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    return fb


FILTERBANK = _linear_filterbank()


def feature_names() -> list[str]:
    """Names of the values returned by :func:`spectral_features`."""
    names = [f"lfcc{i}_{s}" for s in ("mean", "std") for i in range(N_LFCC)]
    names += [f"dlfcc{i}_std" for i in range(N_LFCC)]
    names += [f"band_{lo}_{hi}_{s}" for s in ("mean", "std") for lo, hi in BANDS]
    names += [
        "flatness_mean",
        "flatness_std",
        "hi_flatness_mean",
        "centroid_mean",
        "centroid_std",
        "rolloff95_mean",
        "energy_range_db",
        "pause_ratio",
        "lead_silence_s",
        "trail_silence_s",
        "frame_energy_std",
        "dc_offset",
        "crest_factor_db",
        "clip_ratio",
        "zcr_mean",
        "zcr_std",
        "spectral_flux_mean",
        "spectral_flux_std",
        "duration_s",
    ]
    return names


def spectral_features(x: np.ndarray) -> np.ndarray:
    """Compute the hand-crafted feature vector of one clip.

    Args:
        x: Waveform at 16 kHz.

    Returns:
        A float32 vector aligned with :func:`feature_names`.

    """
    x = x.astype(np.float64)
    if x.size < N_FFT * 2:
        x = np.pad(x, (0, N_FFT * 2 - x.size))
    f, _, z = stft(x, SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    power = np.abs(z) ** 2 + 1e-12
    frame_db = 10 * np.log10(power.sum(axis=0))
    loud = np.percentile(frame_db, 95)
    active = frame_db > loud - 30
    act = power[:, active] if active.sum() >= 3 else power

    lfcc = dct(np.log(FILTERBANK @ act + 1e-12), axis=0, norm="ortho")[:N_LFCC]
    dl = np.diff(lfcc, axis=1) if lfcc.shape[1] > 1 else np.zeros_like(lfcc)
    band_db = []
    for lo, hi in BANDS:
        sel = (f >= lo) & (f < hi)
        band_db.append(
            10 * np.log10(act[sel].sum(axis=0)) - 10 * np.log10(act.sum(axis=0))
        )
    band_db = np.array(band_db)

    geo = np.exp(np.mean(np.log(act), axis=0))
    flat = geo / act.mean(axis=0)
    hi_sel = (f > 4000) & (f < 7000)
    hi_flat = np.exp(np.mean(np.log(act[hi_sel]), axis=0)) / act[hi_sel].mean(axis=0)
    centroid = (f[:, None] * act).sum(axis=0) / act.sum(axis=0)
    cum = np.cumsum(act, axis=0) / act.sum(axis=0)
    rolloff = f[np.argmax(cum > 0.95, axis=0)]
    sec_per_frame = HOP / SAMPLE_RATE
    lead = float(np.argmax(active) * sec_per_frame)
    trail = float(np.argmax(active[::-1]) * sec_per_frame)
    sign = np.sign(x)
    zc = np.abs(np.diff(sign)) > 0
    zcr = zc[: zc.size // HOP * HOP].reshape(-1, HOP).mean(axis=1)
    lp = np.log(power)
    flux = (
        np.sqrt(np.mean(np.diff(lp, axis=1) ** 2, axis=0)) if lp.shape[1] > 1 else [0]
    )
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    vec = np.concatenate(
        [
            lfcc.mean(axis=1),
            lfcc.std(axis=1),
            dl.std(axis=1),
            band_db.mean(axis=1),
            band_db.std(axis=1),
            [
                flat.mean(),
                flat.std(),
                hi_flat.mean(),
                centroid.mean(),
                centroid.std(),
                rolloff.mean(),
                loud - np.percentile(frame_db, 5),
                1 - active.mean(),
                lead,
                trail,
                frame_db.std(),
                x.mean(),
                20 * np.log10(np.abs(x).max() / rms + 1e-12),
                float(np.mean(np.abs(x) > 0.999)),
                zcr.mean(),
                zcr.std(),
                float(np.mean(flux)),
                float(np.std(flux)),
                x.size / SAMPLE_RATE,
            ],
        ]
    )
    return np.nan_to_num(vec).astype(np.float32)
