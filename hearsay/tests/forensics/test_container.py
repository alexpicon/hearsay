# Author: Alex Picon <alexnpc@me.com>
"""Exercise container inspection when optional metadata tools are unavailable."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from forensics import container
from forensics.finding import Context
from forensics.riff import length_grids


def _wav(path: Path) -> Path:
    """Write a known native-rate PCM file independently of external utilities."""
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    return path


def test_missing_metadata_tools_use_riff(tmp_path, monkeypatch) -> None:
    """Missing ffprobe/exiftool leave useful WAV measurements without decoding."""

    def missing(*args, **kwargs):
        raise FileNotFoundError("optional metadata utility is not installed")

    monkeypatch.setattr(container.subprocess, "run", missing)
    path = _wav(tmp_path / "elevenlabs_xtts_clip.wav")
    ctx = Context(path)
    result = container.analyze(path, ctx)
    assert result.features["native_sr"] == 16000
    assert result.features["duration_s"] == 1
    assert result.features["lossy_codec"] == 0
    assert result.features["synth_tool_tag"] == 0
    assert "16000/200" in result.evidence["length_grids"]
    assert ctx._audio is None


@pytest.mark.parametrize("rate", [0, "0", -1, None, "N/A"])
def test_invalid_probe_rate_uses_header(tmp_path, monkeypatch, rate) -> None:
    """Unavailable or malformed probe rates do not override valid RIFF values."""
    probe = {"streams": [{"codec_type": "audio", "sample_rate": rate}]}
    monkeypatch.setattr(container, "probe", lambda path: (probe, {}))
    path = _wav(tmp_path / "clip.wav")
    assert container.analyze(path, Context(path)).features["native_sr"] == 16000


@pytest.mark.parametrize("rate", [0, -1])
def test_no_rate_means_no_grid_claim(rate: int) -> None:
    """Without a usable rate the grid detector abstains instead of dividing."""
    assert length_grids(16000, rate) == []
