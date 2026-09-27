# Author: Alex Picon <alexnpc@me.com>
"""Evidence plots for the HTML report, rendered to inline PNG."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from forensics.audio import power_spectrogram, speech_mask  # noqa: E402

if TYPE_CHECKING:
    from forensics.router import Report

INK, MUTED, ACCENT, ALERT = "#1f2933", "#7b8794", "#2f6fb3", "#c2410c"


def _series(report: Report, expert: str) -> dict:
    """Return the plot series an expert stored in its evidence.

    Args:
        report: Router report.
        expert: Expert name.

    Returns:
        Series dictionary (empty when the expert did not run).

    """
    finding = report.findings.get(expert)
    return finding.evidence.get("series", {}) if finding else {}


def _png(fig: plt.Figure) -> str:
    """Encode a figure as a base64 PNG data URI.

    Args:
        fig: Matplotlib figure.

    Returns:
        Data URI.

    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def evidence_figure(report: Report) -> str | None:
    """Draw spectrogram + pitch, long-term spectrum and voice consistency.

    Args:
        report: Router report with its context.

    Returns:
        PNG data URI, or None without audio context.

    """
    ctx = report.context
    if ctx is None:
        return None
    f, p = power_spectrogram(ctx)
    t = np.arange(p.shape[1]) * 0.01
    fig = plt.figure(figsize=(10, 6.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(grid[0, :])
    db = 10 * np.log10(p)
    lo, hi = np.percentile(db, 5), db.max()
    ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=lo, vmax=hi,
              extent=(0, t[-1] if len(t) else 1, 0, f[-1] / 1000))  # fmt: skip
    mask = speech_mask(ctx)[: len(t)]
    ax.fill_between(t[: len(mask)], 0, 0.25, where=mask, color=ACCENT, alpha=0.9,
                    step="mid", label="speech")  # fmt: skip
    f0 = np.array(_series(report, "prosody").get("f0", []), dtype=float)
    if f0.size:
        tf = np.arange(len(f0)) * 0.01
        ax2 = ax.twinx()
        ax2.plot(tf, np.where(f0 > 0, f0, np.nan), color="#7dd3fc", lw=1.6)
        ax2.set_ylabel("pitch (Hz)", color=MUTED)
        ax2.set_ylim(50, 450)
    ax.set_title(
        "Spectrogram, detected speech (blue strip) and pitch contour", loc="left"
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("kHz")
    ax = fig.add_subplot(grid[1, 0])
    spec = _series(report, "spectral")
    if spec:
        ax.plot(np.array(spec["ltas_f"]) / 1000, spec["ltas_db"], color=INK, lw=1.4)
        bw = report.findings["spectral"].features.get("bandwidth_hz")
        if bw:
            ax.axvline(bw / 1000, color=ALERT, ls="--", lw=1)
            ax.text(bw / 1000, max(spec["ltas_db"]), f" edge {bw:.0f} Hz",
                    color=ALERT, va="top", fontsize=8)  # fmt: skip
    ax.set_title("Long-term speech spectrum", loc="left")
    ax.set_xlabel("kHz")
    ax.set_ylabel("dB")
    ax = fig.add_subplot(grid[1, 1])
    spk = _series(report, "speaker")
    if spk:
        ax.plot(spk["t"], spk["sim"], marker="o", color=ACCENT, lw=1.4)
        ax.set_ylim(min(0.5, min(spk["sim"]) - 0.05), 1.0)
        ax.set_title("Voice similarity to whole clip, per 1 s window", loc="left")
        ax.set_xlabel("window start (s)")
    else:
        level = _series(report, "environment").get("level_db", [])
        ax.plot(np.arange(len(level)) * 0.02, level, color=INK, lw=1)
        ax.set_title("Frame level (dBFS): pauses and noise floor", loc="left")
        ax.set_xlabel("time (s)")
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
    return _png(fig)
