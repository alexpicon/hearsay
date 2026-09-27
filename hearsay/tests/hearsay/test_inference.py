# Author: Alex Picon <alexnpc@me.com>
"""Tests for TSV writing and the default-score policy."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hearsay.inference import write_tsv
from hearsay.train import default_score, operating_points


class TestWriteTsv:
    """Submission file format."""

    def test_header_rows_and_values(self, tmp_path: Path) -> None:
        """Exact header and every valid probability are preserved."""
        rows = [
            {"file": "a.wav", "probability": 1.0},
            {"file": "b.wav", "probability": 0.0},
        ]
        out = tmp_path / "sub.tsv"
        write_tsv(rows, out)
        assert out.read_text().splitlines()[0] == "filename\tcm-score"
        df = pd.read_csv(out, sep="\t")
        assert list(df["cm-score"]) == [1.0, 0.0]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
    def test_invalid_score_preserves_existing_file(
        self, tmp_path: Path, value: float
    ) -> None:
        """Invalid results cannot overwrite a previously valid submission."""
        out = tmp_path / "submission.tsv"
        out.write_text("previous submission\n")
        with pytest.raises(ValueError):
            write_tsv([{"file": "clip.wav", "probability": value}], out)
        assert out.read_text() == "previous submission\n"

    def test_duplicate_names_rejected(self, tmp_path: Path) -> None:
        """A repeated clip identity cannot silently replace another result."""
        rows = [{"file": "a.wav", "probability": 0.3}] * 2
        with pytest.raises(ValueError):
            write_tsv(rows, tmp_path / "submission.tsv")


def test_long_audio_discloses_ssl_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-file duration must not imply SSL analysis past its actual prefix."""
    from types import SimpleNamespace

    from hearsay import inference

    detector = SimpleNamespace(
        thresholds={"analyst": 0.8, "as_coded": 0.2},
        predict=lambda *_: np.array([0.6]),
    )
    monkeypatch.setattr(inference, "get_detector", lambda *_: detector)
    monkeypatch.setattr(inference, "load_audio", lambda *_: np.zeros(14 * 16000))
    monkeypatch.setattr(inference, "features_for", lambda *_: (None, None))
    result = inference.score_file("long.wav", explain=False)
    assert result["duration_s"] == 14
    assert result["ssl_analyzed_duration_s"] == 12
    assert result["ssl_truncated"] is True
    assert result["analysis_coverage"]["spectral"]["end_s"] == 14


def test_directory_ignores_non_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only supported visible audio files become challenge rows."""
    from hearsay import inference

    for name in ["clip.wav", "second.mp3", "README.txt", ".hidden.wav"]:
        (tmp_path / name).touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "third.wav").touch()

    def score_existing(path: Path, *_args: object, **_kw: object) -> dict:
        """Check that nested clips reach the decoder under their real path."""
        assert path.is_file()
        return {"file": path.name, "probability": 0.3}

    monkeypatch.setattr(
        inference,
        "score_file",
        score_existing,
    )
    rows = inference.score_directory(tmp_path, tmp_path / "scores.tsv", threads=1)
    assert {r["file"] for r in rows} == {"clip.wav", "second.mp3", "third.wav"}


class TestDefaultScore:
    """The hedged default lies between both operating points."""

    def test_between_thresholds(self) -> None:
        """Default is inside the interval spanned by the two thresholds."""
        pts = {"analyst": 0.9, "as_coded": 0.1}
        d = default_score(pts)
        assert 0.1 < d < 0.9
        assert d == pytest.approx(0.5)

    def test_operating_points_order(self) -> None:
        """The analyst threshold is never below the as-coded one."""
        import numpy as np

        rng = np.random.default_rng(0)
        y = np.array([0] * 700 + [1] * 300)
        p = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
        pts = operating_points(p, y)
        assert pts["analyst"] >= pts["as_coded"]
