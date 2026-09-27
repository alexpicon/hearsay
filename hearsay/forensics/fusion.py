# Author: Alex Picon <alexnpc@me.com>
"""Evidence fusion: combine expert findings into one synthetic probability.

The fused log-odds are a weighted sum of each expert's log-odds, shrunk by
the expert's own confidence, plus a prior term:

    logit(P) = bias + prior + sum_e  w_e * c_e * logit(s_e)

The weights ``w_e`` are fitted by logistic regression on
labelled data (see ``forensics.fit``); experts that did not run or had no
opinion simply add nothing, which is what makes routing possible. ``prior``
moves the balanced training prior to the deployment prior (about 30%
synthetic in the test set).
"""

from __future__ import annotations

import math

from forensics.calibration import Calibration, logit, sigmoid
from forensics.finding import Finding

TEST_PRIOR_SYNTHETIC = 0.30
# With a false alarm costing 4x a miss, the Bayes decision is "synthetic"
# only when P(synthetic) > 4 / (4 + 1).
DECISION_THRESHOLD = 0.8


def prior_shift(p: float = TEST_PRIOR_SYNTHETIC) -> float:
    """Return the log-odds shift from a balanced prior to ``p``.

    Args:
        p: Deployment prior of the synthetic class.

    Returns:
        Log-odds offset.

    """
    return math.log(p / (1 - p))


def fuse(
    findings: dict[str, Finding],
    calibration: Calibration | None,
    prior: float = TEST_PRIOR_SYNTHETIC,
) -> tuple[float, dict[str, float]]:
    """Fuse the findings that have a score.

    Args:
        findings: Findings keyed by expert.
        calibration: Holds the fusion weights (uniform weights when None).
        prior: Deployment prior of the synthetic class.

    Returns:
        Fused probability and each expert's log-odds contribution.

    """
    weights = calibration.fusion_weights if calibration else {}
    bias = calibration.fusion_bias if calibration else 0.0
    contrib: dict[str, float] = {}
    for name, finding in findings.items():
        if finding.score is None:
            continue
        w = weights.get(name, 0.0 if weights else 1.0)
        contrib[name] = w * finding.confidence * logit(finding.score)
    # Legacy caches may still contain the removed near-silence rule. Such
    # processing evidence is not an independently calibrated likelihood.
    total = bias + prior_shift(prior) + sum(contrib.values())
    return sigmoid(total), contrib


def verdict(p: float) -> str:
    """Translate a fused probability into analyst language.

    Args:
        p: Fused synthetic probability (deployment prior).

    Returns:
        Short verdict.

    """
    if p >= DECISION_THRESHOLD:
        return "likely synthetic"
    if p >= 0.5:
        return "suspicious, below the alert threshold"
    if p >= 0.2:
        return "inconclusive, leaning real"
    return "likely real"
