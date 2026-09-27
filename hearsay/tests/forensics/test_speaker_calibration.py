# Author: Alex Picon <alexnpc@me.com>
"""Backend compatibility checks for current and cached speaker findings."""

import time
from pathlib import Path

import numpy as np
import pytest

from forensics.calibration import Calibration, FeatureModel
from forensics.expert import build_finding
from forensics.finding import Context
from forensics.fit import matrix, rescore


@pytest.fixture
def calibration() -> Calibration:
    """Build a strong WavLM head that must never score MFCC inputs."""
    return Calibration(
        experts={
            "speaker": FeatureModel(
                ["sim_mean"], [0.5], [0.1], [4.0], 1.0, [0.5], [0.1]
            )
        }
    )


@pytest.mark.parametrize("backend", ["mfcc-stats", "unknown", None])
def test_non_wavlm_backend_abstains(calibration: Calibration, backend: str) -> None:
    """Fallback and unidentified measurements cannot inherit WavLM probabilities."""
    ctx = Context(Path("unused.wav"), calibration)
    finding = build_finding(
        ctx,
        "speaker",
        {"sim_mean": 0.9},
        {"backend": backend},
        {"sim_mean": "similarity"},
        1.0,
        time.perf_counter(),
    )
    assert finding.score is None
    assert finding.confidence == 0
    assert not finding.contributions
    assert "abstaining" in finding.summary
    row = {"findings": {"speaker": finding.to_dict()}}
    row["findings"]["speaker"]["score"] = 0.999  # legacy cached wrong-backend score
    restored = rescore(row, calibration)["speaker"]
    assert restored.score is None
    assert np.isnan(matrix([row], "speaker")).all()


def test_wavlm_backend_retains_calibration(calibration: Calibration) -> None:
    """The shipped representation still uses its original calibrated head."""
    finding = build_finding(
        Context(Path("unused.wav"), calibration),
        "speaker",
        {"sim_mean": 0.9},
        {"backend": "wavlm-xvector"},
        {"sim_mean": "similarity"},
        1.0,
        time.perf_counter(),
    )
    assert finding.score > 0.9
    assert finding.confidence == 1.0
