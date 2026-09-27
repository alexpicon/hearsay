# Author: Alex Picon <alexnpc@me.com>
"""Batch survey: header properties shared by most files in a collection.

If more than ``share`` of a batch carries the same sample-count grid, the
grid comes from the collection pipeline (for example silence trimming with a
512-sample hop at 22.05 kHz), not from any individual generator: with at most
about 30% synthetic clips, a synthesis fingerprint cannot cover 60% of files.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from forensics.riff import length_grids, parse_riff


def survey(paths: Iterable[Path], share: float = 0.6) -> dict[str, Any]:
    """Find header properties shared by most of a batch.

    Args:
        paths: Files in the batch.
        share: Minimum fraction for a property to count as shared.

    Returns:
        Batch facts: ``shared_grids`` plus counts for the report.

    """
    grids: Counter[str] = Counter()
    n = 0
    for path in paths:
        if path.suffix.lower() != ".wav":
            continue
        info = parse_riff(path)
        sr = info.get("fmt", {}).get("sr", 16000)
        grids.update(length_grids(info.get("n_samples", 0), sr))
        n += 1
    shared = [g for g, c in grids.items() if n and c / n >= share]
    return {"files": n, "shared_grids": sorted(shared), "grid_counts": dict(grids)}
