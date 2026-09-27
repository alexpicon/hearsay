# Author: Alex Picon <alexnpc@me.com>
"""Named feature sets: which clips are sampled from which corpus.

The HEARSAY test clips are VCTK sentences in an ASVspoof 2019 LA style
channel, so ASVspoof 2019 LA is the core of the training data. DiffSSD adds
modern commercial and diffusion TTS (ElevenLabs, PlayHT, XTTS, OpenVoice ...)
and LJ Speech adds a second bona fide recording condition.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hearsay.augment import ChannelConfig
from hearsay.data import (
    DATA_ROOT,
    diffssd_manifest,
    la19_manifest,
    lj_manifest,
    test_manifest,
    vctk_manifest,
)

logger = logging.getLogger(__name__)
FEATURE_ROOT = DATA_ROOT / "features"


@dataclass(frozen=True)
class SetSpec:
    """How to sample one feature set.

    Attributes:
        name: Set name (also the cache file stem).
        source: ``la19:<partition>``, ``diffssd``, ``lj`` or ``test``.
        bona: Maximum bona fide clips (None keeps all).
        per_system: Maximum synthetic clips per generator.
        augment: Whether the simulated test channel is applied.
        channel: Key of ``CHANNELS`` used when augmenting.
        seed: Offset of the sampling seed (draws other clips than seed 0).

    """

    name: str
    source: str
    bona: int | None = None
    per_system: int | None = None
    augment: bool = True
    channel: str = "default"
    seed: int = 0


CHANNELS = {
    "default": ChannelConfig(),
    # Room tone on most clips, so that recording-room noise is never a class cue.
    "room": ChannelConfig(p_noise=1.0, p_room=0.7),
    # Real recordings that already carry their own room tone.
    "recorded": ChannelConfig(p_noise=0.4, snr_db=(15.0, 35.0), p_room=0.5),
}


SETS = {
    s.name: s
    for s in [
        SetSpec("la_train", "la19:train", None, 400),
        SetSpec("la_dev", "la19:dev", 1500, 300),
        SetSpec("la_eval", "la19:eval", 3000, 300),
        SetSpec("diffssd", "diffssd", None, 120),
        SetSpec("lj", "lj", None, None),
        SetSpec("vctk", "vctk", 4000, None, channel="recorded"),
        SetSpec("vctk_b", "vctk_b", 5000, None, channel="recorded"),
        SetSpec("la_bona_room", "la19:dev", 1500, 0, channel="room", seed=1),
        SetSpec("la_train_room", "la19:train", 0, 250, channel="room", seed=1),
        SetSpec("la_eval_room", "la19:eval", 0, 150, channel="room", seed=1),
        SetSpec("diffssd_room", "diffssd", 0, 100, channel="room", seed=1),
        SetSpec("la_dev_room", "la19:dev", 0, 250, channel="room", seed=2),
        SetSpec("la_eval_room2", "la19:eval", 0, 120, channel="room", seed=2),
        SetSpec("test", "test", None, None, augment=False),
    ]
}


def _load(source: str) -> pd.DataFrame:
    """Full manifest of a source."""
    if source.startswith("la19:"):
        return la19_manifest(source.split(":", 1)[1])
    loaders = {
        "diffssd": diffssd_manifest,
        "lj": lj_manifest,
        "test": test_manifest,
        "vctk": vctk_manifest,
        "vctk_b": lambda: vctk_manifest("ext/vctk_b/vctk_*.parquet"),
    }
    return loaders[source]()


def build_manifest(spec: SetSpec, seed: int = 0) -> pd.DataFrame:
    """Sample the manifest of a set deterministically.

    Args:
        spec: Set specification.
        seed: Sampling seed.

    Returns:
        The manifest ordered for sequential parquet reads.

    """
    seed += spec.seed
    m = _load(spec.source)
    bona = m[m.label == 0]
    if spec.bona is not None and len(bona) > spec.bona:
        bona = bona.sample(spec.bona, random_state=seed)
    spoof = m[m.label == 1]
    if spec.per_system is not None and len(spoof):
        shuffled = spoof.sample(frac=1.0, random_state=seed)
        spoof = shuffled.groupby("system").head(spec.per_system)
    rest = m[m.label == -1]
    out = pd.concat([bona, spoof, rest])
    return out.sort_values(["path", "rg", "row"]).reset_index(drop=True)


def feature_path(
    set_name: str, model_tag: str, shard: int = 0, n_shards: int = 1
) -> Path:
    """Cache location of a set's features.

    Args:
        set_name: Set name.
        model_tag: Short SSL model tag.
        shard: Shard index.
        n_shards: Number of shards.

    Returns:
        The ``.npz`` path.

    """
    suffix = "" if n_shards == 1 else f".{shard}of{n_shards}"
    return FEATURE_ROOT / model_tag / f"{set_name}{suffix}.npz"


def model_tag(model: str) -> str:
    """Short, file-system friendly tag of a Hugging Face model id."""
    return model.split("/")[-1]
