# Author: Alex Picon <alexnpc@me.com>
"""Tests for manifests, audio loading and set sampling."""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from hearsay.audio import load_audio
from hearsay.corpus import SetSpec, build_manifest
from hearsay.data import folder_manifest, iter_audio


def _write(path: Path, seconds: float, sr: int = 16000) -> None:
    """Write a short noise clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = 0.1 * np.random.default_rng(0).standard_normal(int(seconds * sr))
    sf.write(path, x, sr)


class TestFolderManifest:
    """Generator and speaker names come from the folder layout."""

    def test_layout(self, tmp_path: Path) -> None:
        """``<gen>/speaker_x/clip.wav`` gives system gen and speaker speaker_x."""
        _write(tmp_path / "xtts" / "speaker_7" / "a.wav", 1.0)
        _write(tmp_path / "gradtts" / "b.wav", 1.0)
        (tmp_path / "xtts" / ".DS_Store").write_bytes(b"junk")
        m = folder_manifest(tmp_path, "toy", 1).set_index("uid")
        assert len(m) == 2
        assert m.loc["toy/xtts/speaker_7/a.wav", "speaker"] == "speaker_7"
        assert m.loc["toy/gradtts/b.wav", "system"] == "gradtts"

    def test_unlabeled_uses_file_name(self, tmp_path: Path) -> None:
        """Test-set rows are keyed by bare file name with label -1."""
        _write(tmp_path / "HGT0000001.wav", 1.0)
        m = folder_manifest(tmp_path, "test", None)
        assert list(m["uid"]) == ["HGT0000001.wav"]
        assert list(m["label"]) == [-1]


class TestAudio:
    """Loading resamples to 16 kHz mono."""

    def test_resample_and_iter(self, tmp_path: Path) -> None:
        """A 22.05 kHz clip is returned at 16 kHz with the right length."""
        _write(tmp_path / "g" / "a.wav", 2.0, sr=22050)
        assert abs(load_audio(tmp_path / "g" / "a.wav").size - 32000) <= 1
        m = folder_manifest(tmp_path, "toy", 0)
        uid, wav = next(iter_audio(m))
        assert uid == "toy/g/a.wav"
        assert wav is not None and wav.dtype == np.float32

    def test_undecodable_file_yields_none(self, tmp_path: Path) -> None:
        """Garbage bytes produce None instead of an exception."""
        bad = tmp_path / "g" / "bad.wav"
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"not audio at all")
        m = folder_manifest(tmp_path, "toy", 0)
        assert list(iter_audio(m)) == [("toy/g/bad.wav", None)]


class TestSampling:
    """Per-generator caps and determinism."""

    def test_cap_per_system(self, monkeypatch: object, tmp_path: Path) -> None:
        """At most ``per_system`` clips per generator, same draw every time."""
        rows = [
            {
                "uid": f"{s}{i}",
                "source": "x",
                "partition": "x",
                "speaker": "s",
                "system": s,
                "label": 1,
                "path": "p",
                "rg": 0,
                "row": i,
            }
            for s in "AB"
            for i in range(10)
        ]
        rows += [
            {
                "uid": f"b{i}",
                "source": "x",
                "partition": "x",
                "speaker": "s",
                "system": "bonafide",
                "label": 0,
                "path": "p",
                "rg": 0,
                "row": i,
            }
            for i in range(5)
        ]
        import hearsay.corpus as corpus

        monkeypatch.setattr(corpus, "_load", lambda _: pd.DataFrame(rows))
        spec = SetSpec("toy", "toy", None, 3)
        a = build_manifest(spec)
        assert a[a.label == 1].groupby("system").size().to_dict() == {"A": 3, "B": 3}
        assert (a.label == 0).sum() == 5
        assert build_manifest(spec)["uid"].tolist() == a["uid"].tolist()
