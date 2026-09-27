# Author: Alex Picon <alexnpc@me.com>
"""Frozen self-supervised speech front-ends with layer-wise statistics pooling.

The transformer is truncated to its first ``n_layers`` blocks: anti-spoofing
information in wav2vec 2.0 style models peaks in the lower and middle layers,
so the upper half is never computed, which halves CPU time.
"""

import logging
import os
from functools import cache

import numpy as np
import torch
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

logger = logging.getLogger(__name__)
DEFAULT_MODEL = os.environ.get("HEARSAY_SSL", "facebook/wav2vec2-xls-r-300m")
XLS_R_MODEL = "facebook/wav2vec2-xls-r-300m"
XLS_R_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"


def model_revision(name: str) -> str | None:
    """Return the pinned release revision, or an explicit override.

    Args:
        name: Hugging Face model identifier.

    Returns:
        The release revision for XLS-R, an override, or the provider default
        for other experimental front-ends.

    """
    return os.environ.get("HEARSAY_SSL_REVISION") or (
        XLS_R_REVISION if name == XLS_R_MODEL else None
    )


class SSLEmbedder:
    """Mean and standard deviation of every hidden layer of a frozen model."""

    def __init__(self, name: str = DEFAULT_MODEL, n_layers: int = 8) -> None:
        """Load and truncate the model.

        Args:
            name: Hugging Face model id.
            n_layers: Number of transformer blocks to keep.

        """
        self.name = name
        self.revision = model_revision(name)
        config = AutoConfig.from_pretrained(name, revision=self.revision)
        self.group_norm = getattr(config, "feat_extract_norm", "layer") == "group"
        try:
            fe = AutoFeatureExtractor.from_pretrained(name, revision=self.revision)
            self.normalize = bool(getattr(fe, "do_normalize", True))
        except OSError:
            self.normalize = True
        model = AutoModel.from_pretrained(name, revision=self.revision)
        model.encoder.layers = model.encoder.layers[:n_layers]
        self.model = model.eval()
        self.n_layers = n_layers
        self.dim = int(config.hidden_size)

    def _prepare(self, wav: np.ndarray) -> np.ndarray:
        """Apply the model's expected input normalization."""
        if not self.normalize:
            return wav
        return (wav - wav.mean()) / (wav.std() + 1e-7)

    @torch.inference_mode()
    def embed(self, batch: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of equal-length (or cropped) waveforms.

        Args:
            batch: Waveforms at 16 kHz; they are cropped to the shortest one.

        Returns:
            Array of shape ``(B, n_layers + 1, 2, dim)`` in float16 holding the
            per-layer time mean and standard deviation.

        """
        n = min(w.size for w in batch)
        x = np.stack([self._prepare(w[:n]) for w in batch]).astype(np.float32)
        out = self.model(torch.from_numpy(x), output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)
        stats = torch.stack([hs.mean(dim=2), hs.std(dim=2)], dim=2)
        return stats.numpy().astype(np.float16)

    def embed_one(self, wav: np.ndarray, max_seconds: float = 12.0) -> np.ndarray:
        """Embed a single clip of any length.

        Args:
            wav: Waveform at 16 kHz.
            max_seconds: Longer clips are truncated to keep inference bounded.

        Returns:
            Array of shape ``(n_layers + 1, 2, dim)``.

        """
        n = int(max_seconds * 16000)
        wav = wav[:n] if wav.size > n else wav
        if wav.size < 8000:
            wav = np.pad(wav, (0, 8000 - wav.size))
        return self.embed([wav])[0]


@cache
def get_embedder(name: str = DEFAULT_MODEL, n_layers: int = 8) -> SSLEmbedder:
    """Return a process-wide cached embedder.

    Args:
        name: Hugging Face model id.
        n_layers: Number of transformer blocks to keep.

    Returns:
        The embedder.

    """
    logger.info("loading SSL front-end %s (%d layers)", name, n_layers)
    return SSLEmbedder(name, n_layers)
