# Author: Alex Picon <alexnpc@me.com>
"""Feature extraction: load, simulate the channel, embed, cache to disk."""

import logging
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hearsay.audio import crop
from hearsay.augment import ChannelConfig, simulate_channel
from hearsay.data import iter_audio
from hearsay.spectral import feature_names, spectral_features
from hearsay.ssl import get_embedder

logger = logging.getLogger(__name__)


def clip_rng(uid: str, seed: int) -> np.random.Generator:
    """Deterministic per-clip random generator (reproducible augmentation).

    Args:
        uid: Clip identifier.
        seed: Global seed.

    Returns:
        The generator.

    """
    return np.random.default_rng([zlib.crc32(uid.encode()), seed])


def _embed_chunk(
    items: list[tuple[str, np.ndarray]], model: str, n_layers: int, batch: int
) -> dict[str, np.ndarray]:
    """Embed waveforms in length-sorted batches.

    Args:
        items: ``(uid, waveform)`` pairs.
        model: SSL model id.
        n_layers: Transformer blocks to keep.
        batch: Batch size.

    Returns:
        Mapping uid to embedding.

    """
    emb = get_embedder(model, n_layers)
    items = sorted(items, key=lambda t: t[1].size)
    out: dict[str, np.ndarray] = {}
    i = 0
    while i < len(items):
        group = [items[i]]
        # Keep batches whose lengths are within 5% so cropping loses little.
        while (
            len(group) < batch
            and i + len(group) < len(items)
            and items[i + len(group)][1].size <= group[0][1].size * 1.05
        ):
            group.append(items[i + len(group)])
        for (uid, _), e in zip(group, emb.embed([w for _, w in group]), strict=True):
            out[uid] = e
        i += len(group)
    return out


def extract(
    manifest: pd.DataFrame,
    out_path: Path,
    *,
    model: str,
    n_layers: int = 8,
    augment: bool = True,
    seed: int = 0,
    max_seconds: float = 6.0,
    batch: int = 8,
    chunk: int = 256,
    threads: int = 4,
    channel: ChannelConfig | None = None,
) -> Path:
    """Extract SSL and spectral features for a manifest and save them.

    Args:
        manifest: Clips to process.
        out_path: Destination ``.npz`` file.
        model: SSL model id.
        n_layers: Transformer blocks to keep.
        augment: Whether to pass clips through the simulated channel.
        seed: Augmentation seed.
        max_seconds: Crop length (random offset when augmenting).
        batch: SSL batch size.
        chunk: Clips loaded into memory at once.
        threads: Torch intra-op threads.
        channel: Channel configuration (default ``ChannelConfig()``).

    Returns:
        The written path.

    """
    torch.set_num_threads(threads)
    uids: list[str] = []
    ssl_feats: list[np.ndarray] = []
    spec_feats: list[np.ndarray] = []
    pending: list[tuple[str, np.ndarray]] = []
    spec_by_uid: dict[str, np.ndarray] = {}
    t0 = time.time()

    def flush() -> None:
        emb = _embed_chunk(pending, model, n_layers, batch)
        for uid, _ in pending:
            uids.append(uid)
            ssl_feats.append(emb[uid])
            spec_feats.append(spec_by_uid.pop(uid))
        pending.clear()
        rate = len(uids) / (time.time() - t0)
        logger.info(
            "%s: %d/%d clips (%.1f clips/s)",
            out_path.name,
            len(uids),
            len(manifest),
            rate,
        )

    for uid, wav in iter_audio(manifest):
        if wav is None:
            continue
        rng = clip_rng(uid, seed)
        if augment:
            wav = simulate_channel(wav, rng, channel)
        spec_by_uid[uid] = spectral_features(wav)
        wav = crop(wav, max_seconds, rng if augment else None)
        pending.append((uid, wav))
        if len(pending) >= chunk:
            flush()
    if pending:
        flush()
    meta = manifest.set_index("uid").loc[uids].reset_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        uid=np.array(uids),
        ssl=np.stack(ssl_feats) if ssl_feats else np.zeros((0,)),
        spec=np.stack(spec_feats) if spec_feats else np.zeros((0,)),
        spec_names=np.array(feature_names()),
        label=meta["label"].to_numpy(),
        system=meta["system"].to_numpy().astype(str),
        speaker=meta["speaker"].to_numpy().astype(str),
        source=meta["source"].to_numpy().astype(str),
        partition=meta["partition"].to_numpy().astype(str),
        model=np.array(model),
    )
    logger.info("wrote %s (%d clips)", out_path, len(uids))
    return out_path
