# Author: Alex Picon <alexnpc@me.com>
"""CLI integrity checks use small fixtures and no trained models or providers."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hearsay import cli


def _evaluation_files(tmp_path: Path) -> argparse.Namespace:
    scores, key = tmp_path / "scores.tsv", tmp_path / "key.tsv"
    scores.write_text("filename\tcm-score\n001\t0.2\nNA\t0.8\n")
    key.write_text("filename\tcm-label\nNA\tspoof\n001\tbonafide\n")
    return argparse.Namespace(scores=str(scores), key=str(key), spoof_prior=0.3)


def test_evaluate_aligns_literal_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Reordered score/key rows align by exact filename and retain the prior."""
    cli._evaluate(_evaluation_files(tmp_path))
    result = json.loads(capsys.readouterr().out)
    assert result["auc"] == 1
    assert result["min_dcf"] == 0
    assert result["min_dcf_as_coded"] == 0
    assert result["p_spoof"] == 0.3


@pytest.mark.parametrize(
    ("file", "text"),
    [
        ("scores", "filename\tcm-score\n001\t0.2\n"),
        ("scores", "filename\tcm-score\n001\t0.2\nNA\t0.8\nextra\t0.5\n"),
        ("scores", "filename\tcm-score\n001\t0.2\nNA\t0.8\n001\t0.4\n"),
        ("scores", "filename\tcm-score\n001\t0.2\nNA\tnan\n"),
        ("key", "filename\tcm-label\nNA\tspoof\n001\tbonafide\nNA\tspoof\n"),
        ("key", "filename\tcm-label\nNA\tspoof\n001\thuman\n"),
    ],
)
def test_evaluate_never_drops_bad_rows(tmp_path: Path, file: str, text: str) -> None:
    """Evaluation refuses incomplete joins, duplicates and malformed values."""
    args = _evaluation_files(tmp_path)
    Path(getattr(args, file)).write_text(text)
    with pytest.raises(ValueError):
        cli._evaluate(args)


def test_validate_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The release command emits JSON and reports validation errors as exit 2."""
    args = _evaluation_files(tmp_path)
    command = ["hearsay", "validate-submission", "--scores", args.scores]
    monkeypatch.setattr(sys, "argv", [*command, "--template", args.key])
    cli.main()
    assert json.loads(capsys.readouterr().out)["valid"] is True
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "template and/or audio folder" in capsys.readouterr().err


def _cached_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ids: list[str],
    probabilities: list[float],
    template: str,
) -> tuple[argparse.Namespace, list]:
    """Replace feature/model IO with fixtures while exercising the submit handler."""
    features = SimpleNamespace(uid=np.array(ids), ssl=None, spec=None)
    detector = SimpleNamespace(
        ssl_model="fixture", predict=lambda *_: np.array(probabilities)
    )
    module = SimpleNamespace(
        Detector=SimpleNamespace(load=lambda *_: detector),
        Features=SimpleNamespace(load=lambda *_: features),
    )
    writes = []
    monkeypatch.setitem(sys.modules, "hearsay.model", module)
    monkeypatch.setitem(
        sys.modules,
        "hearsay.inference",
        SimpleNamespace(write_tsv=lambda rows, path: writes.append((rows, path))),
    )
    monkeypatch.setattr(cli, "_feature_files", lambda *_: [])
    path = tmp_path / "template.tsv"
    path.write_text(template)
    args = argparse.Namespace(
        set="test", out=str(tmp_path / "out.tsv"), template=str(path), model_path=None
    )
    return args, writes


def test_submit_preserves_exact_names_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached submission sorts exact IDs without renaming or changing scores."""
    args, writes = _cached_submit(
        tmp_path, monkeypatch, ["NA", "001"], [0.8, 0.2], "filename\nNA\n001\n"
    )
    cli._submit(args)
    assert writes == [
        (
            [{"file": "001", "probability": 0.2}, {"file": "NA", "probability": 0.8}],
            Path(args.out),
        )
    ]


@pytest.mark.parametrize(
    ("ids", "scores", "template"),
    [
        (["a", "a"], [0.2, 0.8], "filename\na\n"),
        (["a"], [0.2], "filename\na\nb\n"),
        (["a", "b"], [0.2, 0.8], "filename\na\n"),
        (["a"], [0.2], "filename\na\na\n"),
        (["a"], [np.nan], "filename\na\n"),
        (["a"], [1.2], "filename\na\n"),
        (["a", "b"], [0.2], "filename\na\nb\n"),
    ],
)
def test_submit_fails_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ids: list[str],
    scores: list[float],
    template: str,
) -> None:
    """Missing features cannot become silent defaults or discarded extra IDs."""
    args, writes = _cached_submit(tmp_path, monkeypatch, ids, scores, template)
    with pytest.raises(ValueError):
        cli._submit(args)
    assert writes == []
