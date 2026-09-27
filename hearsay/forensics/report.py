# Author: Alex Picon <alexnpc@me.com>
"""Per-file forensic report as JSON and as a small self-contained HTML page."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from forensics.fusion import DECISION_THRESHOLD
from forensics.plots import evidence_figure

if TYPE_CHECKING:
    from forensics.router import Report

CSS = """
:root{--bg:#fbfbfa;--fg:#1f2933;--muted:#6b7680;--line:#e3e5e8;--card:#fff;
--real:#2f6fb3;--fake:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#15181c;--fg:#e6e8eb;
--muted:#9aa5b1;--line:#2c3238;--card:#1c2024}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;
margin:0;padding:24px 16px}main{max-width:980px;margin:auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:14px 16px}.meter{height:10px;border-radius:5px;
background:linear-gradient(90deg,var(--real),var(--fake));
position:relative;margin:10px 0}
.pin{position:absolute;top:-5px;width:4px;height:20px;background:var(--fg);border-radius:2px}
.thr{position:absolute;top:-3px;width:1px;height:16px;background:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;
padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}img{max-width:100%;border-radius:8px;background:#fff}
.num{font-variant-numeric:tabular-nums;white-space:nowrap}
"""


def to_json(report: Report) -> dict[str, Any]:
    """Serialise a report.

    Args:
        report: Router report.

    Returns:
        JSON-friendly dictionary.

    """
    return {
        "file": report.path.name,
        "synthetic_probability": round(report.probability, 4),
        "verdict": report.verdict,
        "decision_threshold": DECISION_THRESHOLD,
        "expert_logodds": {k: round(v, 3) for k, v in report.contributions.items()},
        "why": reasons(report),
        "trace": report.trace,
        "findings": {k: f.to_dict() for k, f in report.findings.items()},
    }


def reasons(report: Report, top: int = 3) -> list[str]:
    """Explain the verdict with the experts that moved it most.

    Args:
        report: Router report.
        top: How many experts to cite.

    Returns:
        Sentences, strongest first.

    """
    ranked = sorted(report.contributions.items(), key=lambda kv: -abs(kv[1]))
    out = []
    for name, value in ranked[:top]:
        side = "synthetic" if value > 0 else "real"
        expert, _, kind = name.partition(":")
        label = f"{expert} rule" if kind else expert
        summary = report.findings[expert].summary
        out.append(f"{label} ({value:+.2f} log-odds toward {side}): {summary}")
    return out


def _fmt(value: Any) -> str:
    """Format a feature value compactly.

    Args:
        value: Any value.

    Returns:
        Text.

    """
    if isinstance(value, float):
        return f"{value:.3g}"
    return html.escape(str(value))


def to_html(report: Report) -> str:
    """Render the report as one self-contained HTML page.

    Args:
        report: Router report (with its context, for the plots).

    Returns:
        HTML text.

    """
    p = report.probability
    esc = html.escape
    why = (
        "".join(f"<li>{esc(r)}</li>" for r in reasons(report))
        or "<li>no scored evidence</li>"
    )
    trace = "".join(
        f"<tr><td>{esc(s['expert'])}</td><td>{esc(s['action'])}</td>"
        f"<td>{esc(str(s.get('reason', '')))}</td>"
        f"<td class=num>{s.get('posterior_after', '')}</td>"
        f"<td class=num>{s.get('ms', '')}</td></tr>"
        for s in report.trace
    )
    rows = []
    for name, f in report.findings.items():
        score = "no opinion" if f.score is None else f"{f.score:.2f}"
        feats = ", ".join(f"{k}={_fmt(v)}" for k, v in list(f.features.items())[:8])
        rows.append(
            f"<tr><td>{esc(name)}</td><td class=num>{score}</td>"
            f"<td class=num>{f.confidence:.2f}</td><td>{esc(f.summary)}"
            f"<div class=muted>{feats}</div></td></tr>"
        )
    img = evidence_figure(report)
    figure = (
        f'<h2>Evidence plots</h2><img alt="evidence plots" src="{img}">' if img else ""
    )
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Forensic report {esc(report.path.name)}</title><style>{CSS}</style></head>
<body><main><h1>{esc(report.path.name)}</h1>
<div class=muted>HEARSAY forensic report</div>
<div class=card style="margin-top:14px"><b>{esc(report.verdict)}</b>:
synthetic probability <b class=num>{p:.3f}</b>
(alert threshold {DECISION_THRESHOLD}, test prior 30% synthetic)
<div class=meter><div class=thr style="left:{DECISION_THRESHOLD * 100:.0f}%"></div>
<div class=pin style="left:calc({p * 100:.1f}% - 2px)"></div></div>
<ul>{why}</ul></div>
{figure}
<h2>Why each expert ran (router trace)</h2><div class=card><table>
<tr><th>expert</th><th>action</th><th>reason</th><th>P after</th><th>ms</th></tr>
{trace}</table></div>
<h2>Findings</h2><div class=card><table>
<tr><th>expert</th><th>score</th><th>conf.</th><th>finding</th></tr>
{"".join(rows)}</table></div></main></body></html>"""
