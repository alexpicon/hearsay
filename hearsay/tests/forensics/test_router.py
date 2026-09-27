# Author: Alex Picon <alexnpc@me.com>
"""Tests for the router policy, fusion and reports on synthetic audio."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from forensics.calibration import Calibration, FeatureModel
from forensics.finding import Finding
from forensics.fusion import fuse
from forensics.report import to_html, to_json
from forensics.router import Router, RouterConfig


@pytest.fixture(autouse=True)
def _light_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the MFCC speaker backend so tests do not download a model."""
    monkeypatch.setenv("HEARSAY_SPEAKER_BACKEND", "mfcc")
    monkeypatch.delenv("HEARSAY_DEEP_SCORES", raising=False)
    monkeypatch.setenv("HEARSAY_MODEL", "/nonexistent/detector.joblib")


def _speechlike(path: Path, seconds: float = 6.3) -> Path:
    """Write a voiced, syllable-modulated harmonic signal with pauses."""
    sr = 16000
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 20))
    envelope = (np.sin(2 * np.pi * 3 * t) > 0) * (t % 2 < 1.6)
    rng = np.random.default_rng(0)
    y = 0.3 * voice * envelope + 0.003 * rng.standard_normal(t.size)
    sf.write(path, (y / np.abs(y).max() * 0.9).astype(np.float32), sr)
    return path


class TestRouter:
    """Routing decisions and report output."""

    def test_full_policy_runs_all_eligible(self, tmp_path: Path) -> None:
        """Full policy runs every expert except the unavailable deep model."""
        report = Router(None, RouterConfig(policy="full")).run(
            _speechlike(tmp_path / "a.wav")
        )
        ran = {s["expert"] for s in report.trace if s["action"] == "run"}
        assert {"container", "spectral", "compression", "prosody"} <= ran
        assert {"splice", "environment", "speaker"} <= ran
        assert all("reason" in s for s in report.trace)

    def test_routed_policy_skips_when_confident(self, tmp_path: Path) -> None:
        """A confident spectral model stops escalation and says why."""
        cal = Calibration(
            experts={
                "spectral": FeatureModel(
                    features=["flux"], mean=[0.0], scale=[1.0], coef=[0.0],
                    intercept=-8.0, real_median=[0.0], real_spread=[1.0],
                )
            },
            fusion_weights={"spectral": 1.0},
        )  # fmt: skip
        report = Router(cal, RouterConfig()).run(_speechlike(tmp_path / "b.wav"))
        skipped = {s["expert"] for s in report.trace if s["action"] == "skip"}
        assert "speaker" in skipped
        assert report.probability < 0.35

    def test_reports_serialise(self, tmp_path: Path) -> None:
        """JSON is valid and the HTML is self-contained."""
        report = Router(None, RouterConfig(policy="full")).run(
            _speechlike(tmp_path / "c.wav")
        )
        json.dumps(to_json(report))
        page = to_html(report)
        assert page.startswith("<!doctype html>")
        assert "data:image/png;base64," in page
        assert "http://" not in page and "https://" not in page


class TestFusion:
    """Fusion ignores experts without an opinion."""

    def test_none_scores_add_nothing(self) -> None:
        """Only scored experts contribute to the fused log-odds."""
        found = {
            "a": Finding("a", None, 1.0, ""),
            "b": Finding("b", 0.9, 1.0, ""),
        }
        _, contrib = fuse(found, None, prior=0.5)
        assert set(contrib) == {"b"}


class TestProcessingEvidence:
    """Processing cues do not prove that speech was synthesized."""

    @pytest.mark.parametrize("level", [0.0, 1e-5])
    def test_near_zero_segments_are_unscored(
        self, tmp_path: Path, level: float
    ) -> None:
        """Exact and near-zero segments retain measurements without rule odds."""
        path = _speechlike(tmp_path / "d.wav")
        y, sr = sf.read(path, dtype="float32")
        y[sr : sr + 1600] = level
        sf.write(path, y, sr, subtype="FLOAT")
        report = Router(None, RouterConfig(policy="full")).run(path)
        finding = report.findings["environment"]
        assert "environment:rule" not in report.contributions
        assert "rule_logodds" not in finding.evidence
        assert finding.evidence["internal_near_zero_runs"] > 0
        assert "does not establish synthetic speech" in finding.summary

    def test_legacy_rule_is_not_applied(self) -> None:
        """Old cached near-silence bonuses cannot change current probabilities."""
        finding = Finding("environment", None, 0.0, "", evidence={"rule_logodds": 3})
        p, contributions = fuse({"environment": finding}, None, prior=0.5)
        assert p == 0.5
        assert contributions == {}

    def test_file_name_is_not_metadata(self, tmp_path: Path) -> None:
        """A TTS tool name in the file name must not count as a metadata tag."""
        path = _speechlike(tmp_path / "elevenlabs_xtts_clip.wav")
        report = Router(None, RouterConfig(policy="full")).run(path)
        assert report.findings["container"].features["synth_tool_tag"] == 0.0
