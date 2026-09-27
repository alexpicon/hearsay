# Author: Alex Picon <alexnpc@me.com>
"""Tests for RIFF header parsing and sample-count grid detection."""

import struct
from pathlib import Path

import numpy as np
import soundfile as sf

from forensics.riff import length_grids, parse_riff


class TestParseRiff:
    """Header walking and consistency checks."""

    def test_clean_file_has_no_issues(self, tmp_path: Path) -> None:
        """A file written by libsndfile is consistent."""
        path = tmp_path / "a.wav"
        sf.write(path, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
        info = parse_riff(path)
        assert info["issues"] == []
        assert info["n_samples"] == 16000
        assert "fmt " in info["chunks"] and "data" in info["chunks"]

    def test_wrong_riff_size_is_reported(self, tmp_path: Path) -> None:
        """A tampered RIFF size field is flagged."""
        path = tmp_path / "b.wav"
        sf.write(path, np.zeros(1600, dtype=np.float32), 16000, subtype="PCM_16")
        raw = bytearray(path.read_bytes())
        raw[4:8] = struct.pack("<I", 12345)
        path.write_bytes(bytes(raw))
        assert any("RIFF size" in i for i in parse_riff(path)["issues"])

    def test_not_a_wav(self, tmp_path: Path) -> None:
        """Non-RIFF bytes are reported, not crashed on."""
        path = tmp_path / "c.wav"
        path.write_bytes(b"ID3" + b"\x00" * 64)
        assert parse_riff(path)["issues"] == ["not a RIFF/WAVE file"]


class TestLengthGrids:
    """Frame-grid detection for native and resampled lengths."""

    def test_native_vocoder_grid(self) -> None:
        """A 22.05 kHz clip of 100 hops of 256 sits on the 256 grid."""
        assert "22050/256" in length_grids(256 * 100, 22050)

    def test_resampled_trim_grid(self) -> None:
        """A 512-hop 22.05 kHz length resampled to 16 kHz is still detected."""
        n = round(512 * 97 * 16000 / 22050) + 1
        assert "22050/512~" in length_grids(n, 16000)

    def test_random_lengths_rarely_match(self) -> None:
        """Chance alignment with the 512-hop grid stays near 1/hop."""
        rng = np.random.default_rng(0)
        lengths = rng.integers(30000, 90000, 3000)
        hits = np.mean([("22050/512~" in length_grids(int(n), 16000)) for n in lengths])
        assert hits < 0.02
