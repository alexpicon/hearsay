# Author: Alex Picon <alexnpc@me.com>
"""The trained detector bundle: heads, fusion, operating points, metadata."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from hearsay.backend import Fusion, LinearHead, TreeHead, sigmoid
from hearsay.explain import describe_feature, summarize

logger = logging.getLogger(__name__)
_PACKAGE_MODEL = Path(__file__).resolve().parent / "assets" / "detector.joblib"
MODEL_PATH = (
    _PACKAGE_MODEL
    if _PACKAGE_MODEL.is_file()
    else Path(__file__).resolve().parents[1] / "models" / "detector.joblib"
)


@dataclass
class Features:
    """Cached features of a set of clips (see ``hearsay.extract``)."""

    uid: np.ndarray
    ssl: np.ndarray
    spec: np.ndarray
    label: np.ndarray
    system: np.ndarray
    speaker: np.ndarray
    source: np.ndarray
    partition: np.ndarray

    @classmethod
    def load(cls, paths: list[Path]) -> Features:
        """Concatenate one or more ``.npz`` caches.

        Args:
            paths: Cache files.

        Returns:
            The merged features.

        """
        parts = [np.load(p, allow_pickle=False) for p in paths]
        keys = [
            "uid",
            "ssl",
            "spec",
            "label",
            "system",
            "speaker",
            "source",
            "partition",
        ]
        return cls(**{k: np.concatenate([d[k] for d in parts]) for k in keys})

    def subset(self, mask: np.ndarray) -> Features:
        """Rows selected by a boolean mask or index array."""
        return Features(
            **{k: getattr(self, k)[mask] for k in self.__dataclass_fields__}
        )

    def __len__(self) -> int:
        """Return the number of clips."""
        return int(self.uid.size)


@dataclass
class Detector:
    """SSL linear head + hand-crafted tree head, fused by logistic regression.

    Attributes:
        ssl_model: Hugging Face id of the frozen front-end.
        n_layers: Transformer blocks kept in the front-end.
        linear: SSL head.
        tree: Hand-crafted feature head.
        fusion: Fusion of the two heads' logits.
        default_score: Score written for clips that cannot be decoded.
        thresholds: Dev operating points (``analyst`` and ``as_coded``).
        meta: Training data, dev metrics and other provenance.

    """

    ssl_model: str
    n_layers: int
    linear: LinearHead
    tree: TreeHead
    fusion: Fusion
    default_score: float = 0.3
    thresholds: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def head_logits(self, ssl: np.ndarray, spec: np.ndarray) -> np.ndarray:
        """Stacked head logits ``(N, 2)``: SSL head, hand-crafted head."""
        return np.column_stack([self.linear.logit(ssl), self.tree.logit(spec)])

    def predict(self, ssl: np.ndarray, spec: np.ndarray) -> np.ndarray:
        """Probability that each clip is synthetic.

        Args:
            ssl: SSL statistics ``(N, L+1, 2, D)``.
            spec: Hand-crafted features ``(N, F)``.

        Returns:
            Probabilities in [0, 1].

        """
        return sigmoid(self.fusion.logit(self.head_logits(ssl, spec)))

    def explain(self, ssl: np.ndarray, spec: np.ndarray, top: int = 5) -> dict:
        """Evidence for a single clip.

        Args:
            ssl: SSL statistics of one clip ``(1, L+1, 2, D)``.
            spec: Hand-crafted features of one clip ``(1, F)``.
            top: Number of hand-crafted features to report.

        Returns:
            Head probabilities, fusion weights, per-layer SSL contributions
            and the hand-crafted features that pushed the score the most.

        """
        logits = self.head_logits(ssl, spec)[0]
        contrib = self.tree.contributions(spec)[0][:-1]
        order = np.argsort(-np.abs(contrib))[:top]
        shares = self.linear.layer_contributions(ssl)[0]
        layers = {
            f"layer_{k}": round(float(v), 3)
            for k, v in zip(self.linear.layers, shares, strict=True)
        }
        ssl_p, spec_p = float(sigmoid(logits[0])), float(sigmoid(logits[1]))
        p = float(sigmoid(self.fusion.logit(logits[None]))[0])
        exact_weights = dict(zip(self.fusion.names, self.fusion.coef, strict=True))
        return {
            "summary": summarize(p, ssl_p, spec_p, layers, exact_weights),
            "ssl_head_probability": ssl_p,
            "spectral_head_probability": spec_p,
            "fusion_weights": self.fusion.weights(),
            "ssl_layer_logit": layers,
            "spectral_evidence": [
                {
                    "feature": self.tree.names[i],
                    "meaning": describe_feature(self.tree.names[i]),
                    "value": round(float(spec[0, i]), 4),
                    "pushes": "synthetic" if contrib[i] > 0 else "bona fide",
                    "shap_logit": round(float(contrib[i]), 3),
                }
                for i in order
            ],
        }

    def save(self, path: Path = MODEL_PATH) -> Path:
        """Serialize with joblib.

        Args:
            path: Destination.

        Returns:
            The written path.

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.info("saved detector to %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    @staticmethod
    def load(path: Path = MODEL_PATH) -> Detector:
        """Load a serialized detector.

        Args:
            path: Bundle path.

        Returns:
            The detector.

        """
        return joblib.load(path)
