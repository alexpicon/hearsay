# Author: Alex Picon <alexnpc@me.com>
"""Batch reports retain source identity across folders and audio extensions."""

import argparse
import csv
import json

from forensics import cli
from forensics.router import Report, Router


def test_repeated_names_keep_separate_reports(tmp_path, monkeypatch) -> None:
    """Two directories and two formats with the same stem never overwrite."""
    paths = [tmp_path / "a/clip.wav", tmp_path / "b/clip.wav", tmp_path / "a/clip.mp3"]
    for p in paths:
        p.parent.mkdir(exist_ok=True)
        p.touch()
    monkeypatch.setattr(
        Router, "run", lambda self, path: Report(path, 0.5, "uncertain", {}, [], {})
    )
    out = tmp_path / "reports"
    tsv = tmp_path / "nested/scores.tsv"
    cli.analyze(
        argparse.Namespace(
            inputs=[str(tmp_path / "a"), str(tmp_path / "b"), str(paths[0])],
            out=str(out),
            tsv=str(tsv),
            html=True,
            policy="routed",
            deep="never",
            jobs=1,
        )
    )
    records = [json.loads(p.read_text()) for p in out.glob("*.json")]
    assert len(records) == len(list(out.glob("*.html"))) == 3
    assert {r["source_path"] for r in records} == {str(p) for p in paths}
    with tsv.open() as handle:
        scored = list(csv.DictReader(handle, delimiter="\t"))
    assert len(scored) == 3
    assert {r["report_id"] for r in scored} == {r["report_id"] for r in records}
    assert cli.report_ids(paths) == cli.report_ids(list(reversed(paths)))
    assert all((out / (r["report_id"] + ".json")).exists() for r in scored)
