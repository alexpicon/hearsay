# Author: Alex Picon <alexnpc@me.com>
"""Release checks bind scores to filenames and to the actual input bytes."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hearsay.release import compare_scores, fingerprint, input_manifest


def test_comparison_aligns_filenames(tmp_path: Path) -> None:
    """Row order does not matter, but every named score must reproduce."""
    old, fresh = tmp_path / "old.tsv", tmp_path / "fresh.tsv"
    old.write_text("filename\tcm-score\na.wav\t0.1\nb.wav\t0.9\n")
    fresh.write_text("filename\tcm-score\nb.wav\t0.900001\na.wav\t0.1\n")
    result = compare_scores(old, fresh)
    assert result["rows"] == 2
    assert result["max_absolute_difference"] == pytest.approx(1e-6)
    fresh.write_text("filename\tcm-score\nb.wav\t0.5\na.wav\t0.1\n")
    with pytest.raises(ValueError, match="reproduction differs"):
        compare_scores(old, fresh)


def test_input_manifest_binds_bytes_and_coverage(tmp_path: Path) -> None:
    """Nested inputs keep identity, exact digests and disclosed prefix coverage."""
    nested = tmp_path / "audio"
    nested.mkdir()
    path = nested / "long.wav"
    sf.write(path, np.zeros(13 * 16000), 16000)
    rows = input_manifest(tmp_path)
    assert len(rows) == 1
    assert rows[0]["filename"] == "long.wav"
    assert rows[0]["duration_s"] == 13
    assert rows[0]["ssl_truncated"] is True
    assert rows[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    before = fingerprint(path)
    sf.write(path, np.ones(13 * 16000) * 0.25, 16000)
    assert fingerprint(path)["sha256"] != before["sha256"]
