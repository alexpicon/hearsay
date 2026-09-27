# Author: Alex Picon <alexnpc@me.com>
"""Release checks reject malformed files and incomplete filename coverage."""

from pathlib import Path

import numpy as np
import pytest

from hearsay.submission import (
    audio_filenames,
    audio_paths,
    read_key,
    read_scores,
    template_filenames,
    validate_submission,
)


def _tsv(tmp_path: Path, text: str, name: str = "scores.tsv") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_literal_ids_and_placeholder_template(tmp_path: Path) -> None:
    """Numeric-looking and NA filenames survive without pandas coercion."""
    scores = _tsv(tmp_path, "filename\tcm-score\n001\t0.2\nNA\t0.8\n")
    template = _tsv(tmp_path, "filename\tcm-score\nNA\t\n001\t\n", "template.tsv")
    names, values = read_scores(scores)
    assert names == ["001", "NA"]
    np.testing.assert_array_equal(values, [0.2, 0.8])
    assert template_filenames(template) == ["NA", "001"]
    assert validate_submission(scores, template=template)["rows"] == 2


@pytest.mark.parametrize("score", ["nan", "inf", "-inf", "-0.01", "1.01", "", "bad"])
def test_invalid_score_rejected(tmp_path: Path, score: str) -> None:
    """NaN, infinity, nonnumeric and outside-range scores fail validation."""
    path = _tsv(tmp_path, f"filename\tcm-score\na.wav\t{score}\n")
    with pytest.raises(ValueError, match="scores"):
        read_scores(path)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "filename\tcm-score\n",
        "filename\tcm-score\na.wav\t0.1\na.wav\t0.2\n",
        "filename\tcm-score\n\t0.1\n",
        "filename\tcm-score\n a.wav\t0.1\n",
        "filename\tcm-score\na.wav \t0.1\n",
        "filename\tcm-score\na.wav\n",
        "filename\tcm-score\na.wav\t0.1\textra\n",
        "filename\tcm-score\tcm-score\na.wav\t0.1\t0.2\n",
        "filename,cm-score\na.wav,0.1\n",
    ],
)
def test_malformed_tsv_rejected(tmp_path: Path, text: str) -> None:
    """Malformed rows, headers and duplicate IDs are never silently repaired."""
    with pytest.raises(ValueError):
        read_scores(_tsv(tmp_path, text))


@pytest.mark.parametrize("label", ["human", "real", "Spoof", "", "1", "nan"])
def test_unknown_key_label_rejected(tmp_path: Path, label: str) -> None:
    """Only the scorer's two exact key labels are accepted."""
    key = _tsv(tmp_path, f"filename\tcm-label\na.wav\t{label}\n")
    with pytest.raises(ValueError, match="cm-label"):
        read_key(key)


def test_duplicate_key_and_template_rejected(tmp_path: Path) -> None:
    """Duplicate expected IDs cannot disappear during set conversion."""
    key = _tsv(tmp_path, "filename\tcm-label\na\tspoof\na\tbonafide\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_key(key)
    with pytest.raises(ValueError, match="duplicate"):
        template_filenames(key)


def test_exact_coverage_both_sources(tmp_path: Path) -> None:
    """Both coverage sources must independently agree with the release IDs."""
    scores = _tsv(tmp_path, "filename\tcm-score\na.wav\t0\nb.FLAC\t1\n")
    template = _tsv(tmp_path, "filename\nb.FLAC\na.wav\n", "template.tsv")
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "a.wav").touch()
    (audio / "b.FLAC").touch()
    (audio / "README.txt").write_text("Not audio.")
    result = validate_submission(scores, template=template, folder=audio)
    assert result == {
        "valid": True,
        "rows": 2,
        "score_min": 0.0,
        "score_max": 1.0,
        "score_direction": "0=bonafide, 1=synthetic",
        "coverage": {"template": str(template), "folder": str(audio)},
    }
    (audio / "c.wav").touch()
    with pytest.raises(ValueError, match="missing 1"):
        validate_submission(scores, template=template, folder=audio)


@pytest.mark.parametrize("expected", ["a.wav", "a.wav\nb.wav\nc.wav", "a.wav\nc.wav"])
def test_missing_extra_ids_fail(tmp_path: Path, expected: str) -> None:
    """Neither missing IDs nor extra score rows are tolerated."""
    scores = _tsv(tmp_path, "filename\tcm-score\na.wav\t0.2\nb.wav\t0.8\n")
    template = _tsv(tmp_path, f"filename\n{expected}\n", "template.tsv")
    with pytest.raises(ValueError, match="coverage"):
        validate_submission(scores, template=template)


def test_audio_folder_rejects_duplicate_basenames(tmp_path: Path) -> None:
    """Recursive input with colliding filenames cannot identify clips uniquely."""
    (tmp_path / "a.wav").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.wav").touch()
    with pytest.raises(ValueError, match="duplicate"):
        audio_filenames(tmp_path)


def test_audio_paths_preserve_nested_paths_and_ignore_hidden(tmp_path: Path) -> None:
    """Predict and validation share real nested paths and the same hidden exclusions."""
    nested = tmp_path / "nested"
    nested.mkdir()
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    wanted = nested / "a.WAV"
    wanted.touch()
    (hidden / "a.WAV").touch()
    (nested / ".hidden.wav").touch()
    (nested / "README.txt").touch()
    assert audio_paths(tmp_path) == [wanted]
    assert audio_filenames(tmp_path) == ["a.WAV"]


def test_coverage_source_required(tmp_path: Path) -> None:
    """A release check must establish coverage rather than only parse scores."""
    with pytest.raises(ValueError, match="template and/or audio"):
        validate_submission(tmp_path / "scores.tsv")
    with pytest.raises(ValueError, match="no filenames"):
        audio_filenames(tmp_path)
    with pytest.raises(ValueError, match="not a directory"):
        audio_filenames(tmp_path / "missing")
