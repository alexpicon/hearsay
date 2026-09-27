# Author: Alex Picon <alexnpc@me.com>
"""Channel profile of a folder of clips (used to match training to the test).

For every clip: duration, peak, RMS, a noise-floor estimate, a global SNR
estimate (speech power over the power of the quietest 20% of 25 ms frames),
the frequency where the spectrum falls 10 dB below its 6-7 kHz level, the
7.7-7.95 kHz level relative to 6-7 kHz (band-limiting), and the
low-frequency (< 60 Hz) rumble level relative to 300-3000 Hz.
"""

import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from hearsay.audio import SAMPLE_RATE, AudioDecodeError, load_audio

logger = logging.getLogger(__name__)


def clip_profile(path: str) -> dict[str, float | str]:
    """Measure the channel statistics of one clip.

    Args:
        path: Audio file.

    Returns:
        The statistics (``error`` is set when decoding fails).

    """
    try:
        x = load_audio(path).astype(np.float64)
    except AudioDecodeError as err:
        return {"file": Path(path).name, "error": str(err)}
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    power = (frames**2).mean(axis=1) + 1e-14
    db = 10 * np.log10(power)
    noise = power[db <= np.percentile(db, 20)].mean()
    f, psd = welch(x, SAMPLE_RATE, nperseg=2048)
    ref = 10 * np.log10(psd[(f > 6000) & (f < 7000)].mean() + 1e-20)
    psd_db = 10 * np.log10(psd + 1e-20)
    below = np.flatnonzero((f > 6500) & (psd_db < ref - 10))
    speech = 10 * np.log10(psd[(f > 300) & (f < 3000)].mean() + 1e-20)
    return {
        "file": Path(path).name,
        "duration_s": x.size / SAMPLE_RATE,
        "peak": float(np.abs(x).max()),
        "rms_dbfs": float(10 * np.log10(np.mean(x**2) + 1e-14)),
        "noise_floor_dbfs": float(10 * np.log10(noise)),
        "snr_db": float(10 * np.log10(max(power.mean() - noise, 1e-14) / noise)),
        "rolloff_10db_hz": float(f[below[0]]) if below.size else float(SAMPLE_RATE / 2),
        "rumble_db": float(10 * np.log10(psd[f < 60].mean() + 1e-20) - speech),
        "top_band_db": float(
            10 * np.log10(psd[(f > 7700) & (f < 7950)].mean() + 1e-20) - ref
        ),
    }


def profile_folder(folder: Path, workers: int = 8) -> tuple[pd.DataFrame, dict]:
    """Profile every audio file of a folder.

    Args:
        folder: Folder with the clips.
        workers: Worker processes.

    Returns:
        The per-clip table and a percentile summary.

    """
    files = sorted(str(p) for p in folder.iterdir() if p.is_file())
    with Pool(workers) as pool:
        rows = pool.map(clip_profile, files, chunksize=16)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number")
    pct = [5, 25, 50, 75, 95]
    summary = {
        "n_files": len(df),
        "n_errors": int(df["error"].notna().sum()) if "error" in df else 0,
        "percentiles": {
            c: {
                f"p{p}": float(v)
                for p, v in zip(pct, np.nanpercentile(num[c], pct), strict=True)
            }
            for c in num.columns
        },
    }
    return df, summary
