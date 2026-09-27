# Author: Alex Picon <alexnpc@me.com>
"""Verify expert, fusion and router evaluation excludes held-out source clips."""

from types import SimpleNamespace

import numpy as np
import pytest

from forensics import evaluate, fit, replay, validation
from forensics.splits import folds


@pytest.fixture
def training_case(monkeypatch):
    """Small separable data with clip IDs observable at every expert fit."""
    rng = np.random.default_rng(4)
    y = np.tile([0, 1], 40)
    rows = [
        {
            "path": f"clip-{i}.wav",
            "findings": {
                "spectral": {
                    "expert": "spectral",
                    "score": None,
                    "confidence": 1.0,
                    "summary": "",
                    "features": {
                        "flux": float(3 * y[i] + rng.normal(0, 0.3)),
                        "identity": i,
                    },
                }
            },
        }
        for i in range(len(y))
    ]
    selected = {"spectral": ["flux", "identity"]}
    monkeypatch.setattr(fit, "FITTED", selected)
    monkeypatch.setattr(evaluate, "FITTED", selected)
    calls = {"outer": 0, "expert": 0, "allowed": set()}
    original_calibration, original_model = fit.fit_calibration, fit.fit_model

    def expert_fit(x, labels, names):
        assert set(x[:, 1].astype(int)) <= calls["allowed"]
        calls["expert"] += 1
        return original_model(x, labels, names)

    def calibrate(training, labels, note):
        calls["outer"] += 1
        calls["allowed"] = {
            r["findings"]["spectral"]["features"]["identity"] for r in training
        }
        cal = original_calibration(training, labels, note)
        cal.meta["training_paths"] = {r["path"] for r in training}
        return cal

    monkeypatch.setattr(fit, "fit_model", expert_fit)
    monkeypatch.setattr(validation, "fit_calibration", calibrate)
    return rows, y, calls


def test_nested_scores_exclude_outer_rows(training_case, monkeypatch) -> None:
    """Every expert fit and final prediction respects the outer partition."""
    rows, y, calls = training_case
    seen = []
    original = fit.rescore

    def rescore(row, cal):
        assert row["path"] not in cal.meta["training_paths"]
        seen.append(row["path"])
        return original(row, cal)

    monkeypatch.setattr(evaluate, "rescore", rescore)
    result = evaluate.within(rows, y, ["real" if not label else "fake" for label in y])
    assert sorted(seen) == sorted(r["path"] for r in rows)
    assert calls["outer"] == 5
    assert calls["expert"] == 30  # five inner heads plus final head, per outer fold
    assert result["fused"]["auc"] > 0.95
    assert result["validation"]["outer_folds"] == 5
    assert "heldout_expert_omission_auc_drop" in result


def test_router_replay_uses_held_out_calibration(training_case, monkeypatch) -> None:
    """Replay receives the matching held-out calibration, never a deployment fit."""
    rows, y, calls = training_case
    seen = []

    class CachedRouter:
        def __init__(self, cal, config, batch, experts):
            self.cal = cal

        def run(self, path):
            assert str(path) not in self.cal.meta["training_paths"]
            seen.append(str(path))
            return SimpleNamespace(probability=0.5, trace=[])

    monkeypatch.setattr(replay, "Router", CachedRouter)
    result = replay.replay_oof(rows, y)
    assert sorted(seen) == sorted(r["path"] for r in rows)
    assert calls["outer"] == 5
    assert result["validation"]["outer_folds"] == 5


def test_repeated_source_paths_never_cross_folds() -> None:
    """Repeated cached rows stay with their source rather than inflating holdouts."""
    y = np.repeat(np.tile([0, 1], 20), 2)
    groups = np.repeat(np.arange(40), 2)
    tested = []
    for training, held in folds(y, groups):
        assert set(groups[training]).isdisjoint(groups[held])
        tested.extend(held)
    assert sorted(tested) == list(range(len(y)))


def test_insufficient_independent_groups_rejected() -> None:
    """Do not silently replace impossible grouped validation with row splitting."""
    with pytest.raises(ValueError, match="two source groups"):
        folds(np.array([0, 0, 1, 1]), np.array(["real", "real", "fake", "fake"]))
