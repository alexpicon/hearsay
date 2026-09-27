# Author: Alex Picon <alexnpc@me.com>
"""Outer validation folds enclosing all expert and fusion model fitting."""

from collections.abc import Iterator
from typing import Any

import numpy as np

from forensics.calibration import Calibration
from forensics.fit import fit_calibration
from forensics.splits import folds

PROTOCOL = "nested clip-disjoint CV; fusion uses inner out-of-fold expert scores"


def calibration_folds(
    rows: list[dict[str, Any]], y: np.ndarray, *, count: int = 5
) -> Iterator[tuple[np.ndarray, Calibration]]:
    """Fit only outer-training rows and yield their held-out row indices."""
    if len(rows) != len(y):
        raise ValueError("one analysis row is required per label")
    identities = np.array([str(r.get("path", i)) for i, r in enumerate(rows)])
    for fold, (tr, te) in enumerate(folds(y, identities, count=count)):
        training = [rows[i] for i in tr]
        cal = fit_calibration(
            training, y[tr], {"validation": PROTOCOL, "outer_fold": fold}
        )
        yield te, cal
