# Author: Alex Picon <alexnpc@me.com>
"""Command line: ``python -m forensics analyze|ablation``.

Examples:
    python -m forensics analyze clip.mp3 --out reports --html
    python -m forensics analyze data/test --out reports --tsv forensic.tsv -j 16
    python -m forensics ablation

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

AUDIO = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac", ".opus", ".aac", ".webm"}
log = logging.getLogger("forensics")


def _files(inputs: list[str]) -> list[Path]:
    """Expand files and directories into a sorted list of audio files.

    Args:
        inputs: Paths given on the command line.

    Returns:
        Audio files.

    """
    out: list[Path] = []
    for item in map(Path, inputs):
        if item.is_dir():
            out += [
                p for p in item.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO
            ]
        elif item.suffix.lower() in AUDIO:
            out.append(item)
    return sorted({p.resolve() for p in out})


def report_ids(paths: list[Path]) -> dict[Path, str]:
    """Return stable report names for canonical source paths; reject collisions."""
    ids = {
        p: re.sub(r"[^A-Za-z0-9._-]", "_", p.name)[:80]
        + "."
        + hashlib.sha256(str(p.resolve()).encode()).hexdigest()
        for p in paths
    }
    if len(set(ids.values())) != len(ids):
        raise ValueError("report identity collision; no reports written")
    return ids


def _worker(args: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Analyse one file and write its JSON (and HTML) report.

    Args:
        args: (path, batch survey, options).

    Returns:
        Row for the summary TSV.

    """
    from forensics.calibration import Calibration
    from forensics.report import to_html, to_json
    from forensics.router import Router, RouterConfig

    path, batch, opt = args
    cal = Calibration.load()
    cfg = RouterConfig(policy=opt["policy"], deep=opt["deep"])
    report = Router(cal, cfg, batch).run(Path(path))
    out = Path(opt["out"])
    data = to_json(report)
    report_id = opt["report_id"]
    data.update(source_path=path, report_id=report_id)
    (out / f"{report_id}.json").write_text(json.dumps(data, indent=1))
    if opt["html"]:
        (out / f"{report_id}.html").write_text(to_html(report))
    ran = [s["expert"] for s in report.trace if s["action"] == "run"]
    return {
        "filename": Path(path).name,
        "p": report.probability,
        "ran": ran,
        "source_path": path,
        "report_id": report_id,
    }


def analyze(args: argparse.Namespace) -> None:
    """Run the router on every input file.

    Args:
        args: Parsed command line.

    """
    from forensics.survey import survey

    files = _files(args.inputs)
    identities = report_ids(files)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    batch = survey(files)
    log.info(
        "%d files; batch-shared length grids: %s", len(files), batch["shared_grids"]
    )
    opt = {"out": str(out), "html": args.html, "policy": args.policy, "deep": args.deep}
    work = [
        (
            str(p),
            {"shared_grids": batch["shared_grids"]},
            opt | {"report_id": identities[p]},
        )
        for p in files
    ]
    os.environ.setdefault("HEARSAY_TORCH_THREADS", "1" if args.jobs > 1 else "4")
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            rows = list(ex.map(_worker, work, chunksize=4))
    else:
        rows = [_worker(w) for w in work]
    if args.tsv:
        destination = Path(args.tsv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                ["filename", "cm-score", "experts_run", "source_path", "report_id"]
            )
            writer.writerows(
                [
                    r["filename"],
                    f"{r['p']:.6f}",
                    ",".join(r["ran"]),
                    r["source_path"],
                    r["report_id"],
                ]
                for r in rows
            )
    for r in rows[:20]:
        log.info(
            "%s  P(synthetic)=%.3f  ran=%s", r["filename"], r["p"], ",".join(r["ran"])
        )


def main() -> None:
    """Parse the command line and dispatch."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(
        prog="forensics", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse files and write reports")
    a.add_argument("inputs", nargs="+", help="audio files or directories")
    a.add_argument("--out", default="reports", help="report directory")
    a.add_argument("--html", action="store_true", help="also write HTML reports")
    a.add_argument("--policy", choices=["routed", "full"], default="routed")
    a.add_argument(
        "--deep", choices=["always", "uncertain", "never"], default="uncertain"
    )
    a.add_argument("--tsv", help="write filename/cm-score summary TSV")
    a.add_argument("-j", "--jobs", type=int, default=1)
    ablation = sub.add_parser(
        "ablation", help="validate and fit a candidate calibration"
    )
    ablation.add_argument("--p-spoof", type=float, default=0.5)
    ablation.add_argument("--calibration-out", type=Path, default=None)
    args = parser.parse_args()
    if args.cmd == "analyze":
        analyze(args)
    else:
        from forensics.ablation import main as ablation_main

        ablation_main(p_spoof=args.p_spoof, calibration_out=args.calibration_out)


if __name__ == "__main__":
    main()
