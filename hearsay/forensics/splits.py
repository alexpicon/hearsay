# Author: Alex Picon <alexnpc@me.com>
"""Deterministic clip-disjoint folds for forensic calibration and evaluation."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def folds(
    y: np.ndarray, groups: np.ndarray | None = None, *, count: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split both classes, keeping repeated source identities in the same fold."""
    y = np.asarray(y)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("validation requires both binary classes")
    if groups is None:
        available = min(int((y == c).sum()) for c in (0, 1))
    else:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError("one source identity is required per label")
        available = min(len(np.unique(groups[y == c])) for c in (0, 1))
    count = min(count, available)
    if count < 2:
        raise ValueError("validation requires at least two source groups per class")
    splitter = (
        StratifiedKFold(count, shuffle=True, random_state=seed)
        if groups is None
        else StratifiedGroupKFold(count, shuffle=True, random_state=seed)
    )
    result = list(splitter.split(np.zeros(len(y)), y, groups))
    if any(
        len(np.unique(y[tr])) != 2 or len(np.unique(y[te])) != 2 for tr, te in result
    ):
        raise ValueError("source groups cannot form folds containing both classes")
    return result
