# Author: Alex Picon <alexnpc@me.com>
"""Download the ASVspoof 2019 LA evaluation subset used by the ablation.

Fetches one parquet shard (about 490 MB) of the public, keyless Hugging Face
copy of the LA evaluation partition, keeps every bona fide clip and a
deterministic 12% of the spoofs, and writes 16-bit WAV files plus
``meta.json`` under ``_internal/hearsay/asvspoof2019/wav``. The shard is
deleted afterwards.

Usage:
    python -m forensics.prepare
"""

from __future__ import annotations

import io
import json
import logging
import random
import urllib.request
from pathlib import Path

import soundfile as sf

from forensics.datasets import DATA

log = logging.getLogger(__name__)
BASE = "https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
SHARD = f"{BASE}/resolve/main/data/test-00000-of-00009.parquet"
PROTOCOL = f"{BASE}/resolve/main/protocols/ASVspoof2019.LA.cm.eval.trl.txt"
SPOOF_SHARE = 0.12


def _download(url: str, dest: Path) -> Path:
    """Download a file unless it already exists.

    Args:
        url: Source URL.
        dest: Destination path.

    Returns:
        The destination path.

    """
    if not dest.exists():
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def main() -> None:
    """Build the balanced LA19 evaluation subset."""
    import pyarrow.parquet as pq

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DATA / "asvspoof2019"
    out = root / "wav"
    for label in ("bonafide", "spoof"):
        (out / label).mkdir(parents=True, exist_ok=True)
    protocol = _download(PROTOCOL, root / "protocol.txt").read_text().splitlines()
    keys = {line.split()[1]: line.split()[3:5] for line in protocol}
    shard = _download(SHARD, root / "test-00000.parquet")
    rng = random.Random(0)
    meta = []
    batches = pq.ParquetFile(shard).iter_batches(
        batch_size=256, columns=["path", "audio", "label", "notes"]
    )
    for batch in batches:
        for row in batch.to_pylist():
            uid = row["path"].replace(".flac", "")
            attack, label = keys[uid]
            if label == "spoof" and rng.random() > SPOOF_SHARE:
                continue
            y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            sf.write(out / label / f"{uid}.wav", y, sr, subtype="PCM_16")
            meta.append({"uid": uid, "label": label, "attack": attack})
    (out / "meta.json").write_text(json.dumps(meta))
    shard.unlink()
    log.info("wrote %d clips to %s", len(meta), out)


if __name__ == "__main__":
    main()
