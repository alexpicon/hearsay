# Author: Alex Picon <alexnpc@me.com>
"""Low-level WAV (RIFF) header parsing and sample-count grid checks."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

# (native rate, frame hop) pairs used by common neural vocoders and by
# librosa-style silence trimming (hop 512 at 22.05 kHz).
GRIDS: tuple[tuple[int, int], ...] = (
    (22050, 256),
    (22050, 512),
    (22050, 300),
    (24000, 256),
    (24000, 300),
    (24000, 240),
    (16000, 200),
    (16000, 256),
    (16000, 320),
    (44100, 512),
    (48000, 480),
)


def parse_riff(path: Path) -> dict[str, Any]:
    """Walk the RIFF chunks of a WAV file and check header consistency.

    Args:
        path: WAV file.

    Returns:
        Chunk list, sample count, format fields and a list of issues.

    """
    raw = path.read_bytes()
    issues: list[str] = []
    if len(raw) < 12 or raw[:4] not in (b"RIFF", b"RF64") or raw[8:12] != b"WAVE":
        return {"chunks": [], "issues": ["not a RIFF/WAVE file"], "n_samples": 0}
    riff_size = struct.unpack("<I", raw[4:8])[0]
    if riff_size != len(raw) - 8:
        issues.append(f"RIFF size {riff_size} != file size - 8 ({len(raw) - 8})")
    chunks: list[str] = []
    fmt: dict[str, int] = {}
    data_size = 0
    pos = 12
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4].decode("latin-1")
        size = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        chunks.append(cid)
        body = raw[pos + 8 : pos + 8 + size]
        if cid == "fmt " and len(body) >= 16:
            tag, ch, sr, brate, align, bits = struct.unpack("<HHIIHH", body[:16])
            fmt = {"tag": tag, "channels": ch, "sr": sr, "byte_rate": brate}
            fmt |= {"block_align": align, "bits": bits}
            if align != ch * bits // 8:
                issues.append(f"block align {align} != channels*bits/8")
            if brate != sr * align:
                issues.append(f"byte rate {brate} != sample rate * block align")
        elif cid == "data":
            data_size = size
            if pos + 8 + size > len(raw):
                issues.append("data chunk runs past end of file")
                data_size = len(raw) - pos - 8
        pos += 8 + size + (size & 1)
    if pos < len(raw) and raw[pos:].strip(b"\x00"):
        issues.append(f"{len(raw) - pos} trailing bytes after last chunk")
    if "fmt " not in chunks or "data" not in chunks:
        issues.append("missing fmt or data chunk")
    align = fmt.get("block_align", 0) or 1
    return {
        "chunks": chunks,
        "fmt": fmt,
        "n_samples": data_size // align,
        "issues": issues,
    }


def length_grids(n: int, sr: int) -> list[str]:
    """Return the (rate, hop) grids that an ``n``-sample clip sits on.

    A file resampled from a vocoder's native rate keeps the vocoder's frame
    quantisation: ``n * native / sr`` is (almost) an integer multiple of the
    hop. Chance alignment is about 1/hop per grid, so a hit is evidence of a
    frame-based generator *or* of frame-based trimming; the router decides
    which by looking at the whole batch.

    Args:
        n: Number of samples.
        sr: File sample rate.

    Returns:
        Grid labels such as ``"22050/256"``; a trailing ``~`` marks a match
        after resampling (resamplers may add or drop one sample).

    """
    if n <= 0 or sr <= 0:
        return []
    hits = []
    for native, hop in GRIDS:
        if n < 8 * hop * sr / native:
            continue
        if native == sr:
            if n % hop == 0:
                hits.append(f"{native}/{hop}")
            continue
        grid = hop * sr / native
        offset = n - round(n / grid) * grid
        if -1.2 <= offset <= 1.2:
            hits.append(f"{native}/{hop}~")
    return hits
