# Author: Alex Picon <alexnpc@me.com>
"""Replay the router on cached analyses to measure what routing costs/saves."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from forensics.calibration import Calibration
from forensics.finding import Context, Finding
from forensics.fit import rescore
from forensics.fusion import fuse
from forensics.metrics import summary
from forensics.router import Router, RouterConfig
from forensics.validation import PROTOCOL, calibration_folds


def _lookup(finding: Finding) -> Callable[[Path, Context], Finding]:
    """Return an expert function that just hands back a cached finding.

    Args:
        finding: Cached finding.

    Returns:
        Expert-compatible callable.

    """
    return lambda _path, _ctx: finding


def _replay(
    rows: list[dict[str, Any]],
    y: np.ndarray,
    calibrations: list[Calibration | None],
    config: RouterConfig,
    p_spoof: float,
) -> dict[str, Any]:
    """Compare routed decisions with running every expert.

    Args:
        rows: Cached full analyses.
        y: Labels.
        calibrations: Calibration assigned to each row.
        config: Router settings.
        p_spoof: Metric prior, separate from the router's deployment prior.

    Returns:
        Metrics for both modes, experts run per file, compute share and the
        most common reasons for skipping.

    """
    full, routed, runs, ms_full, ms_routed = [], [], [], 0.0, 0.0
    skips: Counter[str] = Counter()
    escalated = 0
    for row, cal in zip(rows, calibrations, strict=True):
        found = rescore(row, cal or Calibration())
        if config.deep == "never":
            found.pop("deep", None)
        full.append(fuse(found, cal, config.prior)[0])
        experts = {name: _lookup(f) for name, f in found.items()}
        router = Router(cal, config, {}, experts)
        rep = router.run(Path(row["path"]))
        routed.append(rep.probability)
        ran = [s["expert"] for s in rep.trace if s["action"] == "run"]
        runs.append(len(ran))
        escalated += any(e in ran for e in ("splice", "speaker", "deep"))
        ms_full += sum(f.elapsed_ms for f in found.values())
        ms_routed += sum(found[e].elapsed_ms for e in ran if e in found)
        for s in rep.trace:
            if s["action"] == "skip":
                skips[f"{s['expert']}: {s['reason'].split(':')[0]}"] += 1
    return {
        "full": summary(y, np.array(full), p_spoof=p_spoof),
        "routed": summary(y, np.array(routed), p_spoof=p_spoof),
        "mean_experts_run": round(float(np.mean(runs)), 2),
        "escalated_share": round(escalated / max(len(rows), 1), 3),
        "compute_share_vs_full": round(ms_routed / max(ms_full, 1e-9), 3),
        "top_skip_reasons": dict(skips.most_common(8)),
    }


def replay(
    rows: list[dict[str, Any]],
    y: np.ndarray,
    cal: Calibration | None,
    config: RouterConfig | None = None,
    *,
    p_spoof: float = 0.5,
) -> dict[str, Any]:
    """Run a fixed-calibration diagnostic; training independence is unverified."""
    result = _replay(
        rows, y, [cal] * len(rows), config or RouterConfig(deep="never"), p_spoof
    )
    result["validation"] = "fixed-calibration diagnostic; may be resubstitution"
    return result


def replay_oof(
    rows: list[dict[str, Any]],
    y: np.ndarray,
    config: RouterConfig | None = None,
    *,
    p_spoof: float = 0.5,
) -> dict[str, Any]:
    """Replay every clip with experts and fusion fit only on other outer folds."""
    if config is not None and config.deep != "never":
        raise ValueError(
            "OOF replay requires deep='never'; deep scores lack fold provenance"
        )
    calibrations: list[Calibration | None] = [None] * len(rows)
    count = 0
    for held, cal in calibration_folds(rows, y):
        count += 1
        for i in held:
            calibrations[i] = cal
    result = _replay(
        rows, y, calibrations, config or RouterConfig(deep="never"), p_spoof
    )
    result["validation"] = {"protocol": PROTOCOL, "outer_folds": count}
    return result
