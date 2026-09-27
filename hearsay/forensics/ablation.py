# Author: Alex Picon <alexnpc@me.com>
"""Measure every expert honestly and write ``results/forensics_ablation.json``.

For each labelled set outer folds hold out clips from all expert and fusion
fitting; the fusion is fit on inner out-of-fold expert predictions. We check
transfer between domains, whether each expert helps the fused score
(leave-one-expert-out), how much work the router saves, fit the deployment
calibration on the test-like sets and apply it to the unlabelled test set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from forensics import datasets
from forensics.calibration import Calibration
from forensics.evaluate import transfer, within
from forensics.extract import CACHE, extract
from forensics.fit import FITTED, fit_calibration, rescore
from forensics.fusion import DECISION_THRESHOLD, fuse, prior_shift
from forensics.replay import replay_oof

log = logging.getLogger(__name__)
RESULTS = Path(__file__).resolve().parents[1] / "results"
SETS = (
    "lj_diffssd",
    "lj_diffssd_trim_channel",
    "la19",
    "la19_channel",
    "la19_trim_channel",
)
DEPLOY_SETS = ("la19_trim_channel", "lj_diffssd_trim_channel")
TRANSFER = (
    ("lj_diffssd", "la19_trim_channel"),
    ("lj_diffssd_trim_channel", "la19_trim_channel"),
    ("la19", "la19_trim_channel"),
    ("la19_channel", "la19_trim_channel"),
    ("la19_trim_channel", "lj_diffssd_trim_channel"),
)
Loaded = tuple[list[dict[str, Any]], np.ndarray, list[str]]


def labelled(name: str, min_duration: float = 0.0) -> Loaded:
    """Return cached analyses, labels and groups for a labelled set.

    Args:
        name: Dataset name.
        min_duration: Keep only clips at least this long (seconds).

    Returns:
        (rows, labels, groups), skipping files whose analysis failed.

    """
    items = datasets.load(name)
    cached = extract(name, [p for p, _, _ in items])
    keep = [
        (cached[str(p)], y, g)
        for p, y, g in items
        if "findings" in cached[str(p)]
        and (cached[str(p)].get("duration") or 0) >= min_duration
    ]
    return [k[0] for k in keep], np.array([k[1] for k in keep]), [k[2] for k in keep]


def score_test(cal: Calibration) -> dict[str, Any]:
    """Apply the deployment calibration to the cached test-set analyses.

    Writes ``results/forensics_test_scores.tsv`` (fused forensic probability
    plus each expert's score) for fusion with the deep detector.

    Args:
        cal: Deployment calibration.

    Returns:
        Summary statistics of the forensic scores on the test set.

    """
    text = (CACHE / "test.jsonl").read_text()
    rows = [json.loads(line) for line in text.splitlines()]
    experts = [*FITTED, "container"]
    lines = ["filename\tforensic_score\t" + "\t".join(experts)]
    probs = []
    for row in sorted(rows, key=lambda r: r["path"]):
        found = rescore(row, cal)
        p = fuse(found, cal)[0]
        probs.append(p)
        cells = []
        for e in experts:
            f = found.get(e)
            cells.append("" if f is None or f.score is None else f"{f.score:.4f}")
        lines.append(f"{Path(row['path']).name}\t{p:.5f}\t" + "\t".join(cells))
    (RESULTS / "forensics_test_scores.tsv").write_text("\n".join(lines) + "\n")
    arr = np.array(probs)
    q = np.percentile(arr, [5, 25, 50, 75, 95]).round(3).tolist()
    return {
        "files": len(rows),
        "quantiles": dict(zip(["p05", "p25", "p50", "p75", "p95"], q, strict=True)),
        "share_above_alert_threshold": round(
            float(np.mean(arr >= DECISION_THRESHOLD)), 3
        ),
        "share_above_0.5": round(float(np.mean(arr >= 0.5)), 3),
        "note": "Unlabelled; about 30% synthetic expected. Scores use the test prior.",
    }


def main(*, p_spoof: float = 0.5, calibration_out: Path | None = None) -> None:
    """Write nested validation results and a candidate calibration for review."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = {name: labelled(name) for name in SETS}
    data["la19_trim_channel_3s"] = labelled("la19_trim_channel", 3.0)
    result: dict[str, Any] = {
        "datasets": {
            n: {"files": len(y), "synthetic": int(y.sum()), "real": int((1 - y).sum())}
            for n, (_, y, _) in data.items()
        },
        "within_set_oof": {
            n: within(r, y, g, p_spoof=p_spoof) for n, (r, y, g) in data.items()
        },
        "transfer": {
            f"{a} -> {b}": transfer(data[a][:2], data[b][:2], a, p_spoof=p_spoof)
            for a, b in TRANSFER
        },
    }
    rows = [r for n in DEPLOY_SETS for r in data[n][0]]
    y = np.concatenate([data[n][1] for n in DEPLOY_SETS])
    cal = fit_calibration(rows, y, {"train": list(DEPLOY_SETS), "files": len(y)})
    destination = calibration_out or RESULTS / "forensics_calibration.candidate.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cal.save(destination)
    rows_la, y_la, _ = data["la19_trim_channel_3s"]
    result["router_replay_la19_trim_channel_3s"] = replay_oof(
        rows_la, y_la, p_spoof=p_spoof
    )
    result["candidate_calibration"] = {
        "path": str(destination),
        "trained_on": list(DEPLOY_SETS),
        "fusion_weights": cal.fusion_weights,
        "expert_cv_auc": cal.meta.get("expert_auc"),
        "prior_shift_logit": round(prior_shift(), 4),
    }
    RESULTS.mkdir(exist_ok=True)
    result["metric_p_spoof"] = p_spoof
    result["test_set_forensic_scores"] = score_test(cal)
    out = RESULTS / "forensics_ablation.json"
    previous = json.loads(out.read_text()) if out.exists() else {}
    for key in ("observations", "test_set_profile"):
        if key in previous:
            result[key] = previous[key]
    out.write_text(json.dumps(result, indent=1))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
