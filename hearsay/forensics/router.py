# Author: Alex Picon <alexnpc@me.com>
"""Agentic router: decide, per file, which forensic experts to run and why.

Policy (every decision is written to the report's ``trace``):

0. Batch survey (once per run). Read every WAV header and find length grids
   or encoder tags shared by most of the batch. Those are properties of the
   collection pipeline, not of individual clips, so the container expert
   ignores them.
1. Triage, always: ``container`` (headers only, no decoding) and
   ``spectral`` (cheap; the bandwidth it measures steers later choices).
2. Conditional core:
   * ``compression`` only if the container is lossy or the spectrum is
     band-limited below the container's Nyquist (a transcoding suspicion).
   * ``prosody`` only with at least 1 s of detected speech.
   * ``environment`` whenever the clip has pauses or lasts 2 s: it is cheap
     and supplies processing evidence (near-zero segments, mains hum) that
     should never be skipped for being "confident".
   * ``deep`` (black-box model) here only when ``deep="always"``.
3. Escalation, only while uncertain. The alert threshold is P = 0.8 (a false
   alarm costs 4x a miss); while the fused probability sits inside the band
   (``low``, ``high``) around it, extra evidence can still change the call,
   so the router runs the remaining eligible experts in order of expected
   value per unit cost (validation AUC gain / typical run time), re-fusing
   after each and stopping as soon as the probability leaves the band.
   Eligibility: ``splice`` needs 1.5 s of audio, ``speaker`` needs 2 s of
   speech (three 1 s windows), ``deep`` needs a configured model. Metadata
   red flags (synthesis tool tags, header or timestamp inconsistencies) force
   ``splice`` regardless of confidence, because tampering with metadata
   usually goes with editing the audio.

``policy="full"`` runs every eligible expert (used to measure the experts and
the router's savings).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forensics import (
    compression,
    container,
    deep,
    environment,
    prosody,
    speaker,
    spectral,
    splice,
)
from forensics.audio import speech_mask
from forensics.calibration import Calibration
from forensics.finding import Context, Finding
from forensics.fusion import TEST_PRIOR_SYNTHETIC, fuse, verdict

log = logging.getLogger(__name__)
EXPERTS: dict[str, Callable[[Path, Context], Finding]] = {
    "container": container.analyze,
    "spectral": spectral.analyze,
    "compression": compression.analyze,
    "prosody": prosody.analyze,
    "environment": environment.analyze,
    "splice": splice.analyze,
    "speaker": speaker.analyze,
    "deep": deep.analyze,
}
COST_MS = {"splice": 120, "speaker": 900, "deep": 400}
FALLBACK_GAIN = {"deep": 0.4, "splice": 0.05, "speaker": 0.05}


@dataclass
class RouterConfig:
    """Router settings.

    Attributes:
        policy: ``"routed"`` (default) or ``"full"``.
        deep: When to consult the black-box model: always, uncertain, never.
        low: Lower edge of the uncertainty band.
        high: Upper edge of the uncertainty band.
        prior: Deployment prior of the synthetic class.

    """

    policy: str = "routed"
    deep: str = "uncertain"
    low: float = 0.35
    high: float = 0.95
    prior: float = TEST_PRIOR_SYNTHETIC


@dataclass
class Report:
    """Result of routing one file.

    Attributes:
        path: Analysed file.
        probability: Fused synthetic probability (deployment prior).
        verdict: Analyst-language verdict.
        findings: Findings keyed by expert.
        trace: Ordered router decisions.
        contributions: Fused log-odds contribution per expert.
        context: The file context (kept for plotting).

    """

    path: Path
    probability: float
    verdict: str
    findings: dict[str, Finding]
    trace: list[dict[str, Any]]
    contributions: dict[str, float]
    context: Context | None = field(default=None, repr=False)


class Router:
    """Runs experts on one file according to the documented policy."""

    def __init__(
        self,
        calibration: Calibration | None,
        config: RouterConfig | None = None,
        batch: dict[str, Any] | None = None,
        experts: dict[str, Callable[[Path, Context], Finding]] | None = None,
    ) -> None:
        """Create a router.

        Args:
            calibration: Fitted models and fusion weights.
            config: Router settings.
            batch: Output of :func:`survey`.
            experts: Expert functions (defaults to :data:`EXPERTS`; replaced
                by cached lookups when replaying routing decisions).

        """
        self.calibration = calibration
        self.config = config or RouterConfig()
        self.batch = batch or {}
        self.experts = experts if experts is not None else EXPERTS

    def _posterior(self, ctx: Context) -> float:
        """Return the current fused probability for a context."""
        return fuse(ctx.findings, self.calibration, self.config.prior)[0]

    def _run(self, ctx: Context, name: str, reason: str, trace: list) -> None:
        """Run one expert and record the step."""
        started = time.perf_counter()
        try:
            ctx.findings[name] = self.experts[name](ctx.path, ctx)
        except Exception as err:  # noqa: BLE001 - one expert must not kill a file
            log.warning("%s failed on %s: %s", name, ctx.path.name, err)
            trace.append({"expert": name, "action": "error", "reason": str(err)})
            return
        trace.append(
            {
                "expert": name,
                "action": "run",
                "reason": reason,
                "posterior_after": round(self._posterior(ctx), 4),
                "ms": round(1000 * (time.perf_counter() - started), 1),
            }
        )

    def _gain(self, name: str) -> float:
        """Return the expected value per millisecond of an escalation expert."""
        meta = self.calibration.meta.get("expert_auc", {}) if self.calibration else {}
        auc = meta.get(name)
        gain = abs(auc - 0.5) if auc is not None else FALLBACK_GAIN[name]
        return gain / COST_MS[name]

    def run(self, path: Path) -> Report:
        """Analyse one file.

        Args:
            path: Audio file.

        Returns:
            The report.

        """
        cfg = self.config
        ctx = Context(path=path, calibration=self.calibration, batch=self.batch)
        trace: list[dict[str, Any]] = []
        skip = lambda n, r: trace.append({"expert": n, "action": "skip", "reason": r})  # noqa: E731
        self._run(ctx, "container", "always: header and metadata triage", trace)
        self._run(ctx, "spectral", "always: bandwidth steers later choices", trace)
        cont = ctx.findings.get("container")
        spec = ctx.findings.get("spectral")
        lossy = bool(cont and cont.features.get("lossy_codec"))
        banded = bool(spec and spectral.is_band_limited(spec.features, ctx.sr))
        if lossy or banded or cfg.policy == "full":
            why = "lossy codec in container" if lossy else "band-limited: transcoding?"
            self._run(
                ctx, "compression", why if cfg.policy != "full" else "full", trace
            )
        else:
            skip("compression", "full-band PCM, no transcoding suspicion")
        speech_s = float(speech_mask(ctx).sum()) / 100
        if speech_s >= 1.0:
            self._run(ctx, "prosody", f"{speech_s:.1f} s of speech available", trace)
        else:
            skip("prosody", f"only {speech_s:.1f} s of speech")
        if ctx.duration >= 2.0 or speech_s < ctx.duration - 0.3:
            why = "cheap processing check (near-zero segments, mains hum)"
            self._run(
                ctx, "environment", why if cfg.policy != "full" else "full", trace
            )
        else:
            skip("environment", "short clip without pauses")
        use_deep = deep.available() and cfg.deep != "never"
        if use_deep and (cfg.deep == "always" or cfg.policy == "full"):
            self._run(ctx, "deep", "deep model set to always run", trace)
        flags = cont and any(
            cont.features.get(k)
            for k in ("synth_tool_tag", "header_issues", "timestamp_issues")
        )
        eligible = {
            "splice": (ctx.duration >= 1.5, "clip shorter than 1.5 s"),
            "speaker": (speech_s >= 2.0, f"{speech_s:.1f} s speech < 2.0 s"),
            "deep": (
                use_deep and "deep" not in ctx.findings,
                "already run" if "deep" in ctx.findings else "no deep model configured",
            ),
        }
        order = sorted(eligible, key=self._gain, reverse=True)
        for name in order:
            ok, why_not = eligible[name]
            if not ok:
                skip(name, why_not)
                continue
            p = self._posterior(ctx)
            uncertain = cfg.low < p < cfg.high
            forced = flags and name == "splice"
            if cfg.policy == "full" or uncertain or forced:
                why = (
                    "metadata red flags"
                    if forced and not uncertain
                    else (f"uncertain: P={p:.2f} inside ({cfg.low}, {cfg.high})")
                )
                self._run(ctx, name, "full" if cfg.policy == "full" else why, trace)
            else:
                skip(name, f"confident: P={p:.2f} outside ({cfg.low}, {cfg.high})")
        prob, contrib = fuse(ctx.findings, self.calibration, cfg.prior)
        return Report(path, prob, verdict(prob), ctx.findings, trace, contrib, ctx)
