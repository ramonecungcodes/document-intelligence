#!/usr/bin/env python3
"""Render the measurements as a page, from the reports on disk and nothing else.

    python -m web.build
    python -m web.build --out web/dashboard.html --reports reports

A generator rather than an application, for the same reason every other stage here
writes a file: the numbers on this page have to be traceable to the run that produced
them. A server reading a database can show a figure nobody can reproduce. This reads
`report.json` and `calibration-*.json`, and anything it cannot find is a missing panel
with a note saying which command would fill it -- never a plausible-looking default.

Nothing is invented. The handoff this borrows its design from is a fictional product at
fictional scale -- 48,210 samples, 4,812 documents a day, a drift alert about a vendor
template. The corpus underneath this page is about 1,400 documents and every number is
one that was measured. Keeping the mockup's scale would have made the design the most
credible thing here and the measurements the least.

Self-contained output: one HTML file, styles inlined, charts drawn as SVG by hand. No
build step, no CDN, no fonts to fetch.
"""
from __future__ import annotations

import argparse
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ----------------------------------------------------------------------- loading

def load(path: str):
    """A report, or None. A missing report is a fact about the run, not an error."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def rate(value, digits=1, suffix="%"):
    return "--" if value is None else f"{value * 100:.{digits}f}{suffix}"


def num(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def esc(text) -> str:
    return html.escape(str(text), quote=True)


# ------------------------------------------------------------------------ charts
# Drawn as SVG here rather than by a chart library. Three reasons, in order: the output
# has to be one file with no network access; the shapes are simple enough that a library
# would be more configuration than drawing; and a hand-drawn axis can be made to say
# what this page needs -- the reliability diagram's diagonal is the whole point of it,
# and no charting default puts it there.

def polyline(points, width, height, colour, fill=None, dash=""):
    if not points:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = 0.0, max(1.0, max(ys))
    span_x = (hi_x - lo_x) or 1.0
    span_y = (hi_y - lo_y) or 1.0

    def place(x, y):
        return (round((x - lo_x) / span_x * width, 2),
                round(height - (y - lo_y) / span_y * height, 2))

    path = " ".join(f"{px},{py}" for px, py in (place(x, y) for x, y in points))
    out = ""
    if fill:
        first = place(*points[0])
        last = place(*points[-1])
        out += (f'<polygon points="{first[0]},{height} {path} {last[0]},{height}" '
                f'fill="{fill}" />')
    stroke = f' stroke-dasharray="{dash}"' if dash else ""
    out += (f'<polyline points="{path}" fill="none" stroke="{colour}" '
            f'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"'
            f'{stroke} />')
    return out


def sparkline(values, colour, width=150, height=30, fill=None):
    if not values:
        return ""
    points = list(enumerate(values))
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    normalised = [(x, (v - lo) / span) for x, v in points]
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'preserveAspectRatio="none" aria-hidden="true">'
            f'{polyline(normalised, width, height, colour, fill)}</svg>')


def reliability_chart(bins, width=520, height=260):
    """Confidence against accuracy, with the diagonal a calibrated model would sit on.

    The diagonal is the reference and everything else is read against it: above the line
    is a model doing better than it claims, below it is a model whose confidence would
    route errors past a floor. Bars are widened by document count so a bin holding one
    document cannot look as authoritative as one holding sixty.
    """
    pad_l, pad_b, pad_t, pad_r = 42, 30, 12, 12
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_b - pad_t
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'role="img" aria-label="Reliability diagram">']

    for tick in range(0, 11, 2):
        y = pad_t + plot_h - (tick / 10) * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
                     f'y2="{y:.1f}" stroke="var(--rule)" stroke-width="1" />')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 3.5:.1f}" text-anchor="end" '
                     f'font-size="9" fill="var(--ink-faint)" '
                     f'font-family="var(--mono)">{tick / 10:.1f}</text>')

    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" '
                 f'y2="{pad_t}" stroke="var(--ink-faint)" stroke-width="1.2" '
                 f'stroke-dasharray="4 3" />')

    total = sum(b["n"] for b in bins) or 1
    slot = plot_w / len(bins)
    for index, row in enumerate(bins):
        if not row["n"]:
            continue
        share = row["n"] / total
        bar_w = max(4.0, min(slot * 0.82, slot * 0.28 + slot * 1.9 * share))
        x = pad_l + slot * index + (slot - bar_w) / 2
        y = pad_t + plot_h - row["accuracy"] * plot_h
        colour = ("var(--pass)" if row["accuracy"] >= row["mean_confidence"]
                  else "var(--fail)")
        bar_h = max(1.0, pad_t + plot_h - y)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                     f'height="{bar_h:.1f}" rx="2" fill="{colour}" '
                     f'fill-opacity=".82" />')
        cx = pad_l + slot * index + slot / 2
        cy = pad_t + plot_h - row["mean_confidence"] * plot_h
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.6" '
                     f'fill="var(--surface)" stroke="var(--ink)" stroke-width="1.4" />')
        parts.append(f'<text x="{cx:.1f}" y="{pad_t + plot_h + 13:.1f}" '
                     f'text-anchor="middle" font-size="8" fill="var(--ink-faint)" '
                     f'font-family="var(--mono)">{row["n"]}</text>')

    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" '
                 f'y2="{pad_t + plot_h}" stroke="var(--border-strong)" '
                 f'stroke-width="1" />')
    parts.append(f'<text x="{pad_l}" y="{height - 4}" font-size="9" '
                 f'fill="var(--ink-faint)" font-family="var(--mono)">'
                 f'confidence 0.0 to 1.0, documents per bin below</text>')
    parts.append("</svg>")
    return "".join(parts)


def coverage_chart(curve, width=520, height=260):
    """Accuracy against coverage, with the random-abstention baseline on the same axes.

    The baseline is flat, and drawing it is the argument: declining documents at random
    leaves accuracy exactly where it started, so the vertical distance between the two
    lines is the entire value of sorting by confidence. A coverage curve shown alone
    always looks like an achievement.
    """
    pad_l, pad_b, pad_t, pad_r = 42, 30, 12, 12
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_b - pad_t
    rows = [r for r in curve if r["accuracy"] is not None]
    if not rows:
        return ""
    lo = min(min(r["accuracy"] for r in rows),
             rows[0]["baseline_accuracy"] or 0) - 0.04
    lo = max(0.0, lo)
    span = (1.0 - lo) or 1.0

    def place(coverage, accuracy):
        return (pad_l + coverage * plot_w,
                pad_t + plot_h - (accuracy - lo) / span * plot_h)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'role="img" aria-label="Coverage against accuracy">']
    for step in range(5):
        value = lo + span * step / 4
        y = pad_t + plot_h - (value - lo) / span * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
                     f'y2="{y:.1f}" stroke="var(--rule)" stroke-width="1" />')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 3.5:.1f}" text-anchor="end" '
                     f'font-size="9" fill="var(--ink-faint)" '
                     f'font-family="var(--mono)">{value:.2f}</text>')

    base = rows[0]["baseline_accuracy"]
    if base is not None:
        y = pad_t + plot_h - (base - lo) / span * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
                     f'y2="{y:.1f}" stroke="var(--ink-faint)" stroke-width="1.4" '
                     f'stroke-dasharray="5 4" />')

    ordered = sorted(rows, key=lambda r: r["coverage"])
    points = [place(r["coverage"], r["accuracy"]) for r in ordered]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts.append(f'<polygon points="{points[0][0]:.1f},{pad_t + plot_h} {path} '
                 f'{points[-1][0]:.1f},{pad_t + plot_h}" fill="var(--accent)" '
                 f'fill-opacity=".10" />')
    parts.append(f'<polyline points="{path}" fill="none" stroke="var(--accent)" '
                 f'stroke-width="2" stroke-linejoin="round" />')
    for row, (x, y) in zip(ordered, points):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" '
                     f'fill="var(--accent)" data-floor="{row["threshold"]}" />')

    parts.append(f'<line id="floor-line" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" '
                 f'y2="{pad_t + plot_h}" stroke="var(--ink)" stroke-width="1.4" />')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" '
                 f'y2="{pad_t + plot_h}" stroke="var(--border-strong)" '
                 f'stroke-width="1" />')
    for step in range(5):
        x = pad_l + plot_w * step / 4
        parts.append(f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" '
                     f'font-size="9" fill="var(--ink-faint)" '
                     f'font-family="var(--mono)">{step * 25}%</text>')
    parts.append(f'<text x="{pad_l}" y="{height - 3}" font-size="9" '
                 f'fill="var(--ink-faint)" font-family="var(--mono)">'
                 f'coverage - share of documents answered</text>')
    parts.append("</svg>")
    return ("".join(parts),
            {"padL": pad_l, "plotW": plot_w, "padT": pad_t, "plotH": plot_h})


# ------------------------------------------------------------------------ pieces

def kpi(label, value, note, delta=None, good=None, spark=None):
    tone = "" if good is None else (" kpi-up" if good else " kpi-down")
    colour = "var(--pass)" if good else "var(--fail)"
    bits = [f'<div class="kpi">',
            f'<div class="kpi-label">{esc(label)}</div>',
            f'<div class="kpi-row"><span class="kpi-value">{esc(value)}</span>']
    if delta:
        bits.append(f'<span class="kpi-delta{tone}">{esc(delta)}</span>')
    bits.append("</div>")
    if spark:
        bits.append(f'<div class="kpi-spark">'
                    f'{sparkline(spark, colour, fill=colour + "1f")}</div>')
    bits.append(f'<div class="kpi-note">{esc(note)}</div></div>')
    return "".join(bits)


def pill(text, tone="pass"):
    return f'<span class="pill pill-{tone}">{esc(text)}</span>'


def missing(title, command, why):
    """A panel with nothing behind it says so, and says what would fill it.

    The alternative is a placeholder, and a placeholder on a page of measurements is
    indistinguishable from a measurement until someone acts on it.
    """
    return (f'<section class="card card-missing">'
            f'<div class="card-head"><h2>{esc(title)}</h2></div>'
            f'<p class="prose">{esc(why)}</p>'
            f'<pre class="command">{esc(command)}</pre></section>')


def signals_panel(data) -> str:
    """What predicts a bad extraction, with the model's self-report in the table.

    The control row is the point of this panel, so it is styled as a row and not as a
    footnote. Read without it, a lift of 0.120 is a number with no scale; read with it,
    it is four times what asking the model was worth.
    """
    if not data or not data.get("pooled"):
        return missing(
            "Signals",
            "python -m eval.cli signals --predictions reports/degraded-full.jsonl "
            "--corpus data/degraded",
            "No signal report was found, so nothing here says which observations "
            "predict a bad extraction.")
    rows = [r for r in data["pooled"] if r.get("available")]
    rows.sort(key=lambda r: -(r["lift"] if r["lift"] is not None else -9))
    body = ""
    for row in rows:
        control = row["signal"].startswith("classifier_")
        lift = row["lift"]
        tone = ("flat" if control else
                "pass" if (lift or 0) >= 0.08 else
                "warn" if (lift or 0) > 0.01 else "fail")
        note = ("the model's own confidence" if control else
                "" if row.get("direction_agrees") is not False
                else "ran opposite to expectation")
        body += (
            f'<tr{" style=\"background:var(--surface-sunken)\"" if control else ""}>'
            f'<td class="mono">{esc(row["signal"])}</td>'
            f'<td class="mono r">{row["available"]}</td>'
            f'<td class="mono r">{num(row.get("rho"))}</td>'
            f'<td><span class="meter"><i style="width:'
            f'{min(100, max(0, (lift or 0) * 700)):.0f}%;background:var(--{tone})">'
            f'</i></span><span class="mono">{num(lift)}</span></td>'
            f'<td class="mono" style="color:var(--ink-faint)">{esc(note)}</td></tr>')
    return (
        '<section class="card">'
        '<div class="card-head"><h2>What predicts a bad extraction</h2>'
        f'<span class="hint">{data["documents"]} degraded documents &middot; '
        f'routing the least promising {1 - data["coverage"]:.0%}</span></div>'
        '<p class="prose">Lift is accuracy over sending the same number of documents '
        'to a person at random. A signal worth wiring up has a lift, not merely a '
        'correlation. The shaded row is the model&rsquo;s own confidence, scored on the '
        'same documents by the same method &mdash; measured separately it would be an '
        'anecdote.</p>'
        '<div class="scroll"><table><thead><tr><th>signal</th><th class="r">n</th>'
        '<th class="r">correlation</th><th>lift over random</th><th>note</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div></section>')


def confound_panel(data) -> str:
    """Confidence against extraction, and why the pooled number is the wrong one."""
    if not data or not data.get("by_truth"):
        return ""
    rows = sorted(data["by_truth"], key=lambda r: -r["mean_confidence"])
    body = "".join(
        f'<tr><td class="mono">{esc(r["truth"])}</td>'
        f'<td class="mono r">{r["documents"]}</td>'
        f'<td class="mono r">{num(r["mean_confidence"])}</td>'
        f'<td class="mono r">{num(r["outcome"])}</td>'
        f'<td class="mono r" style="color:var(--'
        f'{"fail" if r["outcome"] < r["mean_confidence"] else "pass"})">'
        f'{r["outcome"] - r["mean_confidence"]:+.3f}</td></tr>' for r in rows)
    gap = data.get("mean_gap")
    return (
        '<section class="card finding">'
        '<div class="card-head"><h2><span class="dot"></span>'
        'The confidence is real, and about the wrong thing</h2>'
        f'<span class="hint">{data["documents"]} documents</span></div>'
        '<p class="prose">The classifier is well calibrated for classification. '
        'Scored against whether the <em>extraction</em> came back right &mdash; which '
        'is what a floor actually decides &mdash; raising the floor makes the accepted '
        'half worse. Split by type it is a confound, not a broken model: the types it '
        'is surest about are the ones that extract worst.</p>'
        '<div class="scroll"><table><thead><tr><th>type</th><th class="r">n</th>'
        '<th class="r">confidence</th><th class="r">extracted</th>'
        f'<th class="r">gap</th></tr></thead><tbody>{body}</tbody></table></div>'
        f'<p class="prose" style="margin-top:12px">Pooled gap '
        f'<span class="mono">{gap:+.3f}</span> over all types, which averages across '
        f'the variable driving both columns. Read the rows, not the total.</p>'
        '</section>')


def routing_panel(data, label: str) -> str:
    """What the policy accepted, and what each gate caught that nothing else did."""
    if not data:
        return ""
    gates = data.get("gates") or {}
    body = "".join(
        f'<tr><td class="mono">{esc(name)}</td>'
        f'<td class="mono r">{block["fired"]}</td>'
        f'<td class="mono r">{block["only_reason"]}</td>'
        f'<td class="mono r">{num(block["mean_outcome"])}</td></tr>'
        for name, block in sorted(gates.items(), key=lambda kv: -kv[1]["fired"]))
    lift = data.get("lift")
    tone = "pass" if (lift or 0) > 0.05 else "warn" if (lift or 0) > 0 else "fail"
    return (
        '<section class="card">'
        f'<div class="card-head"><h2>Routing &mdash; {esc(label)}</h2>'
        f'<span class="hint">{data["documents"]} documents</span></div>'
        '<div class="readout">'
        f'<div><div class="k">ACCEPTED</div><div class="v">'
        f'{rate(data["coverage"])}</div></div>'
        f'<div><div class="k">ACCURACY ACCEPTED</div><div class="v">'
        f'{rate(data["accuracy_accepted"])}</div></div>'
        f'<div><div class="k">AT RANDOM</div><div class="v" '
        f'style="color:var(--ink-faint)">{rate(data["baseline"])}</div></div>'
        f'<div><div class="k">LIFT</div><div class="v" style="color:var(--{tone})">'
        f'{lift:+.3f}</div></div>'
        f'<div><div class="k">REVIEWED BUT PERFECT</div><div class="v">'
        f'{data["reviewed_but_perfect"]}</div></div>'
        '</div>'
        '<p class="prose" style="margin-top:14px">&ldquo;Reviewed but perfect&rdquo; is '
        'the cost of the policy &mdash; documents a person looked at for nothing. '
        '&ldquo;Alone&rdquo; below is how often a gate was the only reason a document '
        'was routed; a gate that never fires alone changes no decisions.</p>'
        '<div class="scroll"><table><thead><tr><th>gate</th><th class="r">fired</th>'
        '<th class="r">alone</th><th class="r">mean outcome</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div></section>')


def repair_panel(data, label: str) -> str:
    """Both arms, the paired interval, and the warning carried through to the page."""
    if not data or not data.get("arms"):
        return ""
    arms = data["arms"]
    harmful = [n for n, r in arms.items()
               if r.get("documents") and (r.get("net_delta") or 0) < 0]
    body = ""
    for name, row in arms.items():
        if not row.get("documents"):
            continue
        net = row["net_delta"]
        tone = "fail" if net < 0 else "pass"
        body += (
            f'<tr><td class="mono">{esc(name)}'
            f'{" <span style=\"color:var(--ink-faint)\">baseline</span>" if name == "rerun" else ""}'
            f'</td>'
            f'<td class="mono r">{row["documents"]}</td>'
            f'<td class="mono r">{num(row["accuracy_before"])}</td>'
            f'<td class="mono r">{num(row["accuracy_after"])}</td>'
            f'<td class="mono r" style="color:var(--{tone});font-weight:700">'
            f'{net:+.3f}</td>'
            f'<td class="mono r">{row["improved"]}</td>'
            f'<td class="mono r" style="color:var(--fail)">{row["damaged"]}</td>'
            f'<td class="mono r">{row["gates_clear"]}</td></tr>')

    banner = ""
    if harmful:
        banner = (
            '<p class="prose" style="background:var(--fail-tint);color:var(--fail-ink);'
            'padding:10px 12px;border-radius:var(--r);font-weight:600;max-width:none">'
            f'NET-NEGATIVE: {esc(", ".join(harmful))} left documents worse than they '
            'were found. Repair is optional; an arm scoring below zero should be off.'
            '</p>')

    verdict = ""
    for name, pair in (data.get("paired") or {}).items():
        if pair.get("mean") is None:
            continue
        low, high = pair["interval"]
        settled = ("The interval excludes zero." if pair["resolvable"]
                   else "The interval does not exclude zero &mdash; this has been "
                        "observed once, not measured.")
        caveat = ("" if not harmful else
                  " But both arms are net-negative here, so this is "
                  "<strong>less harmful, not helpful</strong>.")
        verdict = (
            f'<p class="prose" style="margin-top:12px">{esc(name)} against the blind '
            f're-run, paired over {pair["documents"]} documents: '
            f'<span class="mono">{pair["mean"]:+.4f}</span>, 95% interval '
            f'<span class="mono">[{low:+.4f}, {high:+.4f}]</span>. Better on '
            f'{pair["better"]}, worse on {pair["worse"]}, tied on {pair["tied"]}. '
            f'{settled}{caveat}</p>')

    return (
        f'<section class="card{" finding" if harmful else ""}">'
        f'<div class="card-head"><h2>Repair &mdash; {esc(label)}</h2>'
        '<span class="hint">scored against the corpus, never against the gates</span>'
        '</div>'
        f'{banner}'
        '<p class="prose">The extractor is sampled, so a second request improves some '
        'documents by luck. The blind arm re-asks the identical question and prices '
        'that; anything the guided arm is worth is the distance between them.</p>'
        '<div class="scroll"><table><thead><tr><th>arm</th><th class="r">n</th>'
        '<th class="r">before</th><th class="r">after</th><th class="r">net</th>'
        '<th class="r">better</th><th class="r">worse</th>'
        f'<th class="r">gates cleared</th></tr></thead><tbody>{body}</tbody></table>'
        f'</div>{verdict}</section>')


# -------------------------------------------------------------------------- page

STYLE = """
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: var(--font);
  background: var(--ground);
  color: var(--ink);
  -webkit-font-smoothing: antialiased;
  font-size: 13px;
  line-height: 1.5;
}
.shell { display: flex; min-height: 100vh; }

/* ---- sidebar: the pipeline, in the order a document moves through it ---- */
.rail {
  width: 236px; flex: 0 0 236px; background: var(--chrome);
  color: var(--chrome-text); display: flex; flex-direction: column;
}
.rail-head {
  height: 58px; display: flex; align-items: center; gap: 10px;
  padding: 0 16px; border-bottom: 1px solid var(--chrome-line);
}
.mark {
  width: 24px; height: 24px; border-radius: 6px; background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: 12px; font-weight: 700;
}
.rail-name {
  font-size: 14px; font-weight: 600; color: var(--chrome-bright);
  letter-spacing: -.2px;
}
.tag {
  margin-left: auto; font-size: 9px; font-family: var(--mono);
  color: var(--chrome-faint); border: 1px solid #2a313d;
  padding: 2px 5px; border-radius: 4px;
}
.rail-body { flex: 1; padding: 14px 10px; overflow-y: auto; }
.rail-group {
  font-size: 9px; letter-spacing: 1.4px; color: var(--chrome-faint);
  padding: 0 10px 7px; font-weight: 700;
}
.rail-item {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px; border-radius: var(--r);
  font-size: 12.5px; color: var(--chrome-text); margin-bottom: 1px;
}
.rail-item.on { background: #1c2029; color: var(--chrome-bright); font-weight: 600; }
.rail-item .n {
  margin-left: auto; font-family: var(--mono); font-size: 10px;
  color: var(--chrome-faint);
}
.rail-foot {
  border-top: 1px solid var(--chrome-line); padding: 12px 16px;
  font-size: 10px; font-family: var(--mono); color: var(--chrome-faint);
  line-height: 1.7;
}

/* ---- main ---- */
.main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.head {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 18px 28px; display: flex; align-items: flex-start; gap: 20px;
}
.head h1 { margin: 0; font-size: 21px; font-weight: 600; letter-spacing: -.4px; }
.head .sub { margin: 3px 0 0; font-size: 12px; color: var(--ink-muted); }
.head .right { margin-left: auto; text-align: right; }
.body { padding: 22px 28px 48px; display: flex; flex-direction: column; gap: 18px; }

/* ---- cards ---- */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 18px 20px; box-shadow: var(--shadow);
}
.card-head {
  display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px;
}
.card h2 { margin: 0; font-size: 13px; font-weight: 600; letter-spacing: -.1px; }
.card .hint {
  margin-left: auto; font-size: 10px; color: var(--ink-faint);
  font-family: var(--mono);
}
.card-missing { border-style: dashed; background: var(--surface-sunken); }
.prose { margin: 0 0 10px; font-size: 12px; color: var(--ink-muted); max-width: 62ch; }
.command {
  margin: 0; font-family: var(--mono); font-size: 11px;
  background: var(--chrome); color: var(--chrome-bright);
  padding: 9px 12px; border-radius: var(--r); overflow-x: auto;
}

/* ---- layout helpers ---- */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 14px; }
.two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
@media (max-width: 1080px) { .two { grid-template-columns: minmax(0, 1fr); } }

.kpi {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 15px 16px 13px; box-shadow: var(--shadow);
}
.kpi-label {
  font-size: 10px; letter-spacing: 1.2px; font-weight: 700;
  color: var(--ink-faint); margin-bottom: 8px;
}
.kpi-row { display: flex; align-items: baseline; gap: 7px; }
.kpi-value {
  font-family: var(--mono); font-size: 26px; font-weight: 700;
  letter-spacing: -1px; font-variant-numeric: tabular-nums;
}
.kpi-delta { font-family: var(--mono); font-size: 11px; font-weight: 700; }
.kpi-up { color: var(--pass); }
.kpi-down { color: var(--fail); }
.kpi-spark { margin: 10px 0 6px; }
.kpi-note { font-size: 10.5px; color: var(--ink-faint); line-height: 1.4; }

/* ---- the floor control ---- */
.floor {
  display: flex; align-items: center; gap: 14px; margin-bottom: 16px;
  padding: 12px 14px; background: var(--surface-sunken);
  border: 1px solid var(--border); border-radius: var(--r);
}
.floor label {
  font-size: 10px; letter-spacing: 1.2px; font-weight: 700; color: var(--ink-faint);
}
.floor input[type=range] { flex: 1; accent-color: var(--accent); min-width: 110px; }
.floor .value {
  font-family: var(--mono); font-size: 17px; font-weight: 700;
  font-variant-numeric: tabular-nums; min-width: 52px; text-align: right;
}
.readout { display: flex; flex-wrap: wrap; gap: 22px; margin-top: 14px; }
.readout div { min-width: 92px; }
.readout .k {
  font-size: 9.5px; letter-spacing: 1.1px; font-weight: 700; color: var(--ink-faint);
}
.readout .v {
  font-family: var(--mono); font-size: 18px; font-weight: 700;
  font-variant-numeric: tabular-nums;
}

/* ---- tables ---- */
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th {
  text-align: left; font-size: 9.5px; letter-spacing: 1.1px; font-weight: 700;
  color: var(--ink-faint); padding: 0 10px 8px; white-space: nowrap;
  border-bottom: 1px solid var(--border);
}
th.r, td.r { text-align: right; }
td {
  padding: 9px 10px; border-bottom: 1px solid var(--rule);
  font-variant-numeric: tabular-nums;
}
tbody tr:last-child td { border-bottom: none; }
td.mono, th.mono { font-family: var(--mono); }
.meter {
  display: inline-block; width: 84px; height: 5px; border-radius: 3px;
  background: var(--rule); overflow: hidden; vertical-align: middle;
  margin-right: 8px;
}
.meter i { display: block; height: 100%; border-radius: 3px; }

.pill {
  display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 9px;
  border-radius: var(--r-pill); white-space: nowrap;
}
.pill-pass { background: var(--pass-tint); color: var(--pass-ink); }
.pill-warn { background: var(--warn-tint); color: var(--warn-ink); }
.pill-fail { background: var(--fail-tint); color: var(--fail-ink); }
.pill-flat { background: var(--accent-tint); color: var(--accent-press); }

/* ---- the finding card, where the mockup put a drift alert ---- */
.finding { border-left: 3px solid var(--fail); }
.finding .dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--fail);
  display: inline-block; margin-right: 7px; vertical-align: middle;
}
.finding table { margin-top: 12px; }

.legend {
  display: flex; flex-wrap: wrap; gap: 16px; margin-top: 12px;
  font-size: 10.5px; color: var(--ink-faint);
}
.legend span { display: flex; align-items: center; gap: 6px; }
.swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.dashed {
  width: 14px; border-top: 2px dashed var(--ink-faint); display: inline-block;
}
.foot {
  font-size: 10.5px; color: var(--ink-faint); font-family: var(--mono);
  line-height: 1.8;
}
"""

SCRIPT = """
(function () {
  var data = window.__COVERAGE__;
  if (!data) return;
  var slider = document.getElementById('floor');
  var value = document.getElementById('floor-value');
  var line = document.getElementById('floor-line');
  var geo = data.geometry;
  var out = {
    coverage: document.getElementById('out-coverage'),
    accuracy: document.getElementById('out-accuracy'),
    errors: document.getElementById('out-errors'),
    caught: document.getElementById('out-caught'),
    lift: document.getElementById('out-lift')
  };

  function nearest(floor) {
    var best = data.curve[0], gap = Infinity;
    for (var i = 0; i < data.curve.length; i++) {
      var d = Math.abs(data.curve[i].threshold - floor);
      if (d < gap) { gap = d; best = data.curve[i]; }
    }
    return best;
  }

  function paint() {
    var floor = Number(slider.value) / 100;
    var row = nearest(floor);
    value.textContent = row.threshold.toFixed(2);
    out.coverage.textContent = row.coverage === null
      ? '--' : (row.coverage * 100).toFixed(1) + '%';
    out.accuracy.textContent = row.accuracy === null
      ? '--' : (row.accuracy * 100).toFixed(1) + '%';
    out.errors.textContent = row.errors;
    out.caught.textContent = row.errors_caught;
    // The number the whole panel exists for: what sorting by confidence bought over
    // declining the same count at random. Zero here means the floor is pure lost
    // coverage, and that has to be as legible as a good result.
    var lift = (row.accuracy === null || row.baseline_accuracy === null)
      ? null : row.accuracy - row.baseline_accuracy;
    out.lift.textContent = lift === null
      ? '--' : (lift >= 0 ? '+' : '') + (lift * 100).toFixed(1) + ' pts';
    out.lift.style.color = lift === null ? 'var(--ink-faint)'
      : (lift > 0.001 ? 'var(--pass)' : 'var(--ink-muted)');
    if (line && row.coverage !== null) {
      var x = geo.padL + row.coverage * geo.plotW;
      line.setAttribute('x1', x); line.setAttribute('x2', x);
    }
  }

  slider.addEventListener('input', paint);
  paint();
})();
"""


def build(reports_dir: str, out_path: str) -> str:
    # The cascade's own calibration when it exists, because the cascade is what runs.
    # Falling back to the primary's is better than an empty page and worse than the
    # truth, so the page says which one it is showing rather than leaving the reader to
    # assume the number governs the pipeline.
    cascade = load(os.path.join(reports_dir, "calibration-cascade-template.json"))
    design = cascade or load(os.path.join(reports_dir,
                                          "calibration-dit-template.json"))
    measured_on = "cascade" if cascade else "dit"
    source = load(os.path.join(reports_dir, "calibration-dit-source.json"))
    score = load(os.path.join(reports_dir, "report.json"))
    run = load(os.path.join(reports_dir, "v1-predicted-type.run.json"))
    signals = load(os.path.join(reports_dir, "signals-degraded.json"))
    extraction = load(os.path.join(reports_dir, "calibration-extraction.json"))
    routing = [("degraded corpus",
                load(os.path.join(reports_dir, "routing-degraded.json"))),
               ("clean corpus",
                load(os.path.join(reports_dir, "routing-clean.json")))]
    repairs = [("degraded corpus",
                load(os.path.join(reports_dir, "repair-degraded-v2.json"))
                or load(os.path.join(reports_dir, "repair-degraded.json"))),
               ("clean corpus",
                load(os.path.join(reports_dir, "repair-clean-63.json")))]

    parts = []

    # -------------------------------------------------------------- KPI row
    overall = (score or {}).get("overall", {})
    point = (design or {}).get("operating_point")
    cards = []
    if design:
        target = design.get("target_accuracy", 0.99)
        cards.append(kpi(
            "COVERAGE AT FLOOR",
            rate(point["coverage"]) if point else "none",
            (f"floor {point['threshold']:.2f}, held to {target:.0%} accuracy"
             if point else f"no floor reaches {target:.0%}"),
            good=bool(point)))
        cards.append(kpi(
            "ERRORS THROUGH",
            str(point["errors"]) if point else "--",
            (f"of {design['documents']} decisions; "
             f"{point['errors_caught']} declined" if point else "no floor qualifies"),
            good=bool(point and point["errors"] == 0)))
        gap = design.get("mean_gap")
        cards.append(kpi(
            "CONFIDENCE GAP",
            (f"{gap:+.3f}" if gap is not None else "--"),
            ("overconfident - errors would pass a floor" if (gap or 0) > 0
             else "underconfident - the safe direction"),
            good=(gap is not None and gap <= 0)))
        cards.append(kpi("ECE", num(design.get("ece")),
                         "mean gap between stated and observed"))
    if overall:
        cards.append(kpi("FIELD ACCURACY", rate(overall.get("field_accuracy")),
                         f"{overall.get('fields_graded', 0):,} fields graded"))
        cards.append(kpi("EXACT MATCH", rate(overall.get("field_exact")),
                         f"{overall.get('scored', 0)} documents scored"))
    if cards:
        parts.append(f'<div class="kpis">{"".join(cards)}</div>')

    # ------------------------------------------------------- the floor panel
    if design and design.get("curve"):
        chart, geometry = coverage_chart(design["curve"])
        start = int(round((point["threshold"] if point else 0.9) * 100))
        readout = "".join(
            f'<div><div class="k">{label}</div>'
            f'<div class="v" id="{ident}">--</div></div>'
            for label, ident in (("COVERAGE", "out-coverage"),
                                 ("ACCURACY", "out-accuracy"),
                                 ("ERRORS THROUGH", "out-errors"),
                                 ("ERRORS CAUGHT", "out-caught"),
                                 ("OVER RANDOM", "out-lift")))
        parts.append(
            '<section class="card">'
            '<div class="card-head"><h2>Where the confidence floor goes</h2>'
            f'<span class="hint">design holdout &middot; {measured_on} &middot; '
            f'{design["documents"]} decisions</span></div>'
            '<p class="prose">Answer at or above the floor, send the rest to a person. '
            'The dashed line is what declining the same number of documents at random '
            'would give &mdash; the overall accuracy, unchanged. A floor is worth only '
            'the distance between the two.</p>'
            '<div class="floor"><label for="floor">FLOOR</label>'
            f'<input type="range" id="floor" min="0" max="100" step="5" '
            f'value="{start}" aria-label="confidence floor">'
            '<span class="value" id="floor-value">--</span></div>'
            f'{chart}'
            '<div class="legend">'
            '<span><i class="swatch" style="background:var(--accent)"></i>'
            'accuracy on answered</span>'
            '<span><i class="dashed"></i>random abstention at the same coverage</span>'
            '</div>'
            f'<div class="readout">{readout}</div>'
            '</section>')
        payload = {"curve": design["curve"], "geometry": geometry}
        parts.append(f'<script>window.__COVERAGE__ = {json.dumps(payload)};</script>')

    # ---------------------------------------------- reliability + the finding
    right = ""
    if design and source:
        design_point = design.get("operating_point")
        source_point = source.get("operating_point")
        rows = [
            ("held out by source document", source, source_point),
            ("held out by page design", design, design_point),
        ]
        body = "".join(
            f'<tr><td>{esc(label)}</td>'
            f'<td class="mono r">{num(d.get("accuracy"))}</td>'
            f'<td class="mono r">{p["threshold"]:.2f}</td>'
            f'<td class="mono r">{rate(p["coverage"])}</td></tr>'
            if p else
            f'<tr><td>{esc(label)}</td>'
            f'<td class="mono r">{num(d.get("accuracy"))}</td>'
            f'<td class="mono r">none</td><td class="mono r">--</td></tr>'
            for label, d, p in rows)
        right = (
            '<section class="card finding">'
            '<div class="card-head"><h2><span class="dot"></span>'
            'The holdout decides the number</h2>'
            '<span class="hint">measured, not estimated</span></div>'
            '<p class="prose">Holding out whole documents leaves the model free to '
            'recognise a page design it trained on. Holding out a whole design does '
            'not, and it is the split that predicts a vendor template nobody has seen. '
            'The same model, the same corpus, two splits:</p>'
            '<div class="scroll"><table><thead><tr><th>split</th>'
            '<th class="r">accuracy</th><th class="r">floor</th>'
            '<th class="r">coverage</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>'
            '<p class="prose" style="margin-top:12px">The manifest carried '
            '<code>0.90</code>. On the honest split <code>0.85</code> answers more '
            'documents with the same zero errors, so five points of coverage were '
            'being given away for nothing.</p>'
            '</section>')

    left = ""
    if design and design.get("bins"):
        left = ('<section class="card">'
                '<div class="card-head"><h2>Reliability</h2>'
                '<span class="hint">bars sized by document count</span></div>'
                '<p class="prose">A calibrated model sits on the diagonal: its 0.8 bin '
                'is right 80% of the time. Bars above the dot are a model doing better '
                'than it claims.</p>'
                f'{reliability_chart(design["bins"])}'
                '<div class="legend">'
                '<span><i class="swatch" style="background:var(--pass)"></i>'
                'accuracy above stated confidence</span>'
                '<span><i class="swatch" style="background:var(--fail)"></i>'
                'below &mdash; the direction that leaks errors</span></div>'
                '</section>')
    if left or right:
        parts.append(f'<div class="two">{left}{right}</div>')

    # --------------------------------------------------------- by degradation
    if design and len(design.get("profiles", [])) > 1:
        body = ""
        for row in design["profiles"]:
            point_p = row.get("operating_point")
            accuracy = row.get("accuracy") or 0
            tone = "pass" if accuracy >= 0.85 else "warn" if accuracy >= 0.7 else "fail"
            floor = f"{point_p['threshold']:.2f}" if point_p else "none"
            cover = rate(point_p["coverage"]) if point_p else "--"
            body += (
                f'<tr><td class="mono">{esc(row["profile"])}</td>'
                f'<td class="mono r">{row["documents"]}</td>'
                f'<td><span class="meter"><i style="width:{accuracy * 100:.0f}%;'
                f'background:var(--{tone})"></i></span>'
                f'<span class="mono">{num(row.get("accuracy"))}</span></td>'
                f'<td class="mono r">{num(row.get("mean_confidence"))}</td>'
                f'<td class="mono r">{num(row.get("ece"))}</td>'
                f'<td class="mono r">{floor}</td>'
                f'<td class="mono r">{cover}</td></tr>')
        parts.append(
            '<section class="card">'
            '<div class="card-head"><h2>By degradation</h2>'
            '<span class="hint">one floor over all four costs the clean ones</span>'
            '</div>'
            '<p class="prose">The floors diverge. A single global floor is set by the '
            'worst profile and charges every other profile for it, which is the '
            'argument for routing on the degradation the normalizer already '
            'reports.</p>'
            '<div class="scroll"><table><thead><tr><th>profile</th>'
            '<th class="r">n</th><th>accuracy</th><th class="r">confidence</th>'
            '<th class="r">ECE</th><th class="r">floor</th>'
            '<th class="r">coverage</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div></section>')
    elif not design:
        parts.append(missing(
            "Calibration",
            "python -m eval.cli calibrate --report reports/dit-template.json",
            "No calibration report was found, so every panel about confidence is "
            "absent rather than estimated."))

    # ------------------------------------------------------ field accuracy
    if score:
        rows = []
        for slice_row in score.get("slices", []):
            if slice_row.get("dimension") != "doc_type":
                continue
            for field in slice_row.get("fields", []):
                rows.append((slice_row["slice"], field))
        rows.sort(key=lambda r: (r[1].get("accuracy") is None,
                                 r[1].get("accuracy") or 0))
        body = ""
        for slice_name, field in rows[:18]:
            accuracy = field.get("accuracy")
            tone = ("fail" if (accuracy or 0) < 0.85 else
                    "warn" if (accuracy or 0) < 0.95 else "pass")
            action = ("Retrain" if tone == "fail" else
                      "Watch" if tone == "warn" else "Healthy")
            notes = []
            if field.get("spurious"):
                notes.append(f"{field['spurious']} invented")
            if field.get("missing"):
                notes.append(f"{field['missing']} missing")
            body += (
                f'<tr><td class="mono">{esc(field["field"])}</td>'
                f'<td class="mono" style="color:var(--ink-faint)">'
                f'{esc(slice_name)}</td>'
                f'<td class="mono r">{field.get("n", 0)}</td>'
                f'<td><span class="meter"><i style="width:'
                f'{(accuracy or 0) * 100:.0f}%;background:var(--{tone})"></i></span>'
                f'<span class="mono">{num(accuracy)}</span></td>'
                f'<td class="mono r">{num(field.get("exact"))}</td>'
                f'<td class="mono" style="color:var(--ink-faint)">'
                f'{esc(", ".join(notes))}</td>'
                f'<td class="r">{pill(action, tone)}</td></tr>')
        parts.append(
            '<section class="card">'
            '<div class="card-head"><h2>Field accuracy</h2>'
            f'<span class="hint">weakest 18 of {len(rows)} &middot; '
            f'{overall.get("fields_graded", 0):,} graded</span></div>'
            '<p class="prose">Sorted worst first, because the mean hides exactly the '
            'fields worth acting on. "Invented" is a value the model supplied where '
            'the document had none &mdash; the more expensive error, because a blank '
            'gets looked at and a confident wrong answer does not.</p>'
            '<div class="scroll"><table><thead><tr><th>field</th><th>type</th>'
            '<th class="r">n</th><th>accuracy</th><th class="r">exact</th>'
            '<th>note</th><th class="r">action</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div></section>')
    else:
        parts.append(missing(
            "Field accuracy",
            "python -m eval.cli score --predictions reports/predictions.jsonl",
            "No score report was found."))

    # The findings that reframe everything above them, in the order they were made.
    parts.append(confound_panel(extraction))
    parts.append(signals_panel(signals))
    for label, data in routing:
        parts.append(routing_panel(data, label))
    for label, data in repairs:
        parts.append(repair_panel(data, label))
    parts = [p for p in parts if p]

    # ------------------------------------------------------------ provenance
    provenance = []
    if run:
        extractor = (run.get("extractor") or {}).get("settings", {})
        usage = run.get("usage") or {}
        provenance = [
            f"extractor   {extractor.get('model', '--')}",
            f"documents   {run.get('documents', '--')}   "
            f"failed {run.get('failed', '--')}",
            f"type from   {run.get('type_from', '--')}",
            f"latency     {usage.get('latency_s', 0) / 3600:.1f} h over "
            f"{usage.get('calls', '--')} calls",
        ]
    if score and score.get("provenance"):
        provenance.append(
            f"predictions {score['provenance'].get('predictions', '--')}")
    parts.append(
        '<section class="card"><div class="card-head">'
        '<h2>What produced these numbers</h2></div>'
        '<div class="foot">' +
        "<br>".join(esc(line) for line in provenance) +
        '</div><p class="prose" style="margin-top:12px">Every figure on this page is '
        'read from a report file in <code>reports/</code>. Nothing is stored in this '
        'page except what those files say, and a panel with no report behind it says '
        'so rather than showing a placeholder.</p></section>')

    # ------------------------------------------------------------------ rail
    stages = [("Splitter", "every_page"), ("Normalizer", "doctr"),
              ("Classifier", "cascade"), ("Extractor", "openai"),
              ("Validator", "5 rules")]
    rail_stages = "".join(
        f'<div class="rail-item">{esc(name)}<span class="n">{esc(plugin)}</span></div>'
        for name, plugin in stages)

    with open(os.path.join(HERE, "tokens.css"), encoding="utf-8") as handle:
        tokens = handle.read()

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibration</title>
<style>{tokens}{STYLE}</style>
</head>
<body>
<div class="shell">
  <nav class="rail">
    <div class="rail-head">
      <div class="mark">D</div>
      <div class="rail-name">DocumentIntelligence</div>
      <div class="tag">IDP</div>
    </div>
    <div class="rail-body">
      <div class="rail-group">PIPELINE</div>
      {rail_stages}
      <div class="rail-group" style="margin-top:16px">MEASUREMENT</div>
      <div class="rail-item on">Calibration</div>
      <div class="rail-item">Extraction</div>
      <div class="rail-item">Validation</div>
      <div class="rail-item">Splitting</div>
    </div>
    <div class="rail-foot">generated by<br>python -m web.build</div>
  </nav>
  <main class="main">
    <header class="head">
      <div>
        <h1>Calibration</h1>
        <p class="sub">Is the confidence real, and where does the floor go</p>
      </div>
      <div class="right">
        {pill('design holdout', 'flat')}
      </div>
    </header>
    <div class="body">
      {''.join(parts)}
    </div>
  </main>
</div>
<script>{SCRIPT}</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="web.build", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", default=os.path.join(ROOT, "reports"))
    parser.add_argument("--out", default=os.path.join(HERE, "dashboard.html"))
    args = parser.parse_args(argv)
    path = build(args.reports, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
