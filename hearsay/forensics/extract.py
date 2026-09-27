# Author: Alex Picon <alexnpc@me.com>
"""Run every expert on a set of files in parallel and cache the findings."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from forensics.datasets import DATA

log = logging.getLogger(__name__)
CACHE = DATA / "forensics_cache"


def _init_worker() -> None:
    """Keep each worker single-threaded so 16 workers share 16 cores."""
    os.environ.setdefault("HEARSAY_TORCH_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    logging.getLogger().setLevel(logging.ERROR)


def _analyze_full(args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """Analyse one file with every eligible expert (no calibration).

    Args:
        args: (path, batch survey).

    Returns:
        Serialised findings and trace.

    """
    from forensics.router import Router, RouterConfig

    path, batch = args
    router = Router(None, RouterConfig(policy="full", deep="never"), batch)
    try:
        report = router.run(Path(path))
    except Exception as err:  # noqa: BLE001 - keep the batch going
        return {"path": path, "error": str(err)}
    return {
        "path": path,
        "duration": report.context.duration if report.context else None,
        "findings": {k: f.to_dict() for k, f in report.findings.items()},
        "trace": report.trace,
    }


def extract(
    name: str,
    paths: list[Path],
    batch: dict[str, Any] | None = None,
    jobs: int = 16,
) -> dict[str, dict[str, Any]]:
    """Return cached findings for ``paths``, computing the missing ones.

    Args:
        name: Cache name (one JSONL file per dataset).
        paths: Files to analyse.
        batch: Batch survey passed to the router.
        jobs: Worker processes.

    Returns:
        Mapping from path string to its serialised analysis.

    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{name}.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            row = json.loads(line)
            done[row["path"]] = row
    todo = [str(p) for p in paths if str(p) not in done]
    if todo:
        log.info("%s: analysing %d files with %d workers", name, len(todo), jobs)
        with (
            ProcessPoolExecutor(jobs, initializer=_init_worker) as ex,
            cache_file.open("a", encoding="utf-8") as fh,
        ):
            args = [(p, batch or {}) for p in todo]
            for i, row in enumerate(ex.map(_analyze_full, args, chunksize=4)):
                fh.write(json.dumps(row) + "\n")
                done[row["path"]] = row
                if (i + 1) % 200 == 0:
                    log.info("%s: %d/%d", name, i + 1, len(todo))
    return {str(p): done[str(p)] for p in paths}


def _rerun_one(args: tuple[str, str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Re-run one expert on one file (uncalibrated).

    Args:
        args: (path, expert name, batch survey).

    Returns:
        (path, serialised finding).

    """
    from forensics.finding import Context
    from forensics.router import EXPERTS

    path, expert, batch = args
    ctx = Context(path=Path(path), batch=batch)
    return path, EXPERTS[expert](Path(path), ctx).to_dict()


def refresh(
    name: str,
    expert: str,
    batch: dict[str, Any] | None = None,
    jobs: int = 16,
) -> int:
    """Recompute one expert's findings in a cache file after a code change.

    Only rows where the router originally ran (or tried to run) the expert
    are refreshed, so routing eligibility is preserved.

    Args:
        name: Cache name.
        expert: Expert to recompute.
        batch: Batch survey passed to the expert.
        jobs: Worker processes.

    Returns:
        Number of refreshed rows.

    """
    cache_file = CACHE / f"{name}.jsonl"
    rows = [json.loads(line) for line in cache_file.read_text().splitlines()]
    todo = [
        (r["path"], expert, batch or {})
        for r in rows
        if "findings" in r
        and any(
            s["expert"] == expert and s["action"] in ("run", "error")
            for s in r.get("trace", [])
        )
    ]
    with ProcessPoolExecutor(jobs, initializer=_init_worker) as ex:
        fresh = dict(ex.map(_rerun_one, todo, chunksize=16))
    for row in rows:
        if row["path"] in fresh:
            row["findings"][expert] = fresh[row["path"]]
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(cache_file)
    return len(fresh)
