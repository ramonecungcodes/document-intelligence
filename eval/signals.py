"""Which signals predict a bad extraction, and how much routing on one would buy.

`route/features.py` produces the signals; this scores them. They are separate because
the question "what can be observed about this document" and the question "does observing
it help" have different answers, and conflating them is how a feature set grows without
anyone noticing that half of it is inert.

Two numbers per signal, and the second is the one to act on.

**Rank correlation** with the outcome, computed on ranks rather than values because
none of these are linearly related to field accuracy and most are badly scaled -- word
counts run to the thousands and confidences to one. Ranks also mean a single ruined
page cannot drag a correlation on its own. Reported with the direction that was expected
in advance, so a signal that comes out backwards shows up as a surprise instead of
being quietly re-read as confirmation.

**Lift at a coverage** -- if the least promising documents by this signal were sent to
a person, how much better is what remains than sending the same number at random? This
is the only figure that answers whether a signal is worth wiring up. A correlation of
0.3 that buys half a point of accuracy is a correlation worth knowing about and not
worth routing on.

Signals are ranked to a percentile before the coverage curve is drawn, which puts every
one of them -- a confidence in 0..1, a word count in the hundreds, an error count in
single digits -- on the same axis and makes them comparable to each other and to the
classifier's confidence. The cost is that calibration error is meaningless for a ranked
signal, since a percentile is uniform by construction. So ECE is not reported here. A
signal is being asked whether it *orders* documents correctly, not whether its value
means anything, and ordering is all routing needs.

Everything is computed within a document type as well as pooled. Pooling alone already
produced one wrong conclusion in this project: against extraction, the classifier's
confidence is anti-correlated overall and flat-to-positive inside every type, because
the types that extract worst are the ones it is surest about. Any signal here can have
the same confound and the per-type rows are how it would be seen.
"""
from __future__ import annotations

from collections import defaultdict

from eval.calibration import CalibrationScore
from route.features import EXPECTED, NAMES


def _ranks(values: list) -> list:
    """Average ranks, so ties do not manufacture an ordering that is not there.

    Competition ranking would put an arbitrary one of twenty documents that all scored
    zero validator errors ahead of the rest, and a correlation computed on that measures
    the sort order of the input file.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            out[order[position]] = shared
        index = stop + 1
    return out


def spearman(xs: list, ys: list):
    """Rank correlation. None when there is nothing to correlate.

    Returns None rather than zero for a constant signal. Zero is a real finding -- this
    signal is unrelated to the outcome -- and a signal that never varies has not been
    tested at all. Reporting the second as the first would retire a question that was
    never asked.
    """
    if len(xs) < 3:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    top = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    left = sum((a - mx) ** 2 for a in rx) ** 0.5
    right = sum((b - my) ** 2 for b in ry) ** 0.5
    if not left or not right:
        return None
    return round(top / (left * right), 4)


def _percentiles(values: list, direction: int) -> list:
    """Values to 0..1, oriented so higher always means "expect a better extraction".

    Orientation is applied from the direction declared in `route.features.EXPECTED`
    rather than from whichever way the data happens to lean. Flipping a signal to
    whichever sign correlates better is how a noise variable becomes a finding: given
    ten signals, one will point the right way by chance, and letting the data choose
    the direction hides that it was chosen.
    """
    ranked = _ranks(values if direction >= 0 else [-v for v in values])
    n = len(ranked)
    return [(r - 0.5) / n for r in ranked]


def score_signal(name: str, pairs: list, at_coverage: float = 0.8) -> dict:
    """One signal against the outcome. `pairs` is [(value, outcome, profile, truth)]."""
    usable = [p for p in pairs if p[0] is not None]
    row = {
        "signal": name,
        "available": len(usable),
        "missing": len(pairs) - len(usable),
        "expected_direction": EXPECTED.get(name, 0),
        "rho": None,
        "direction_agrees": None,
        "coverage": at_coverage,
        "accuracy_at_coverage": None,
        "baseline": None,
        "lift": None,
    }
    if len(usable) < 3:
        return row

    values = [p[0] for p in usable]
    outcomes = [p[1] for p in usable]
    row["rho"] = spearman(values, outcomes)
    if row["rho"] is not None and EXPECTED.get(name):
        row["direction_agrees"] = (row["rho"] * EXPECTED[name]) > 0

    # Routing on this signal alone, at one stated coverage. The curve is available in
    # full from the returned score, but a table needs one comparable number and the
    # coverage has to be fixed for the comparison to mean anything -- a signal
    # evaluated at its own best coverage will always look better than one evaluated at
    # a fixed one.
    ranked = _percentiles(values, EXPECTED.get(name, 1) or 1)
    score = CalibrationScore(outcome_of="extraction")
    for pseudo, (_value, outcome, profile, truth) in zip(ranked, usable):
        score.add(pseudo, outcome, profile, truth=truth)
    curve = score.curve()
    if curve:
        row["baseline"] = curve[0]["baseline_accuracy"]
        closest = min(curve, key=lambda r: abs((r["coverage"] or 0) - at_coverage))
        row["actual_coverage"] = closest["coverage"]
        row["accuracy_at_coverage"] = closest["accuracy"]
        if closest["accuracy"] is not None and row["baseline"] is not None:
            row["lift"] = round(closest["accuracy"] - row["baseline"], 4)
    return row


def report(rows: list, at_coverage: float = 0.8) -> dict:
    """`rows` is [{signals: {...}, outcome: float, profile: str, truth: str}].

    Pooled and per type. The per-type tables are not decoration: the one conclusion
    this project has already had to overturn came from reading a pooled number whose
    sign was set by a confound.
    """
    def collect(subset):
        out = []
        for name in NAMES:
            pairs = [(r["signals"].get(name), r["outcome"], r.get("profile", "clean"),
                      r.get("truth", "")) for r in subset]
            out.append(score_signal(name, pairs, at_coverage))
        return out

    by_truth = defaultdict(list)
    by_profile = defaultdict(list)
    for row in rows:
        by_truth[row.get("truth") or "unlabelled"].append(row)
        by_profile[row.get("profile") or "clean"].append(row)

    return {
        "documents": len(rows),
        "coverage": at_coverage,
        "pooled": collect(rows),
        "by_truth": {k: collect(v) for k, v in sorted(by_truth.items()) if len(v) >= 8},
        "by_profile": {k: collect(v) for k, v in sorted(by_profile.items())
                       if len(v) >= 8},
    }


def _fmt(value, width=8, digits=3):
    return f"{'--':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def _table(rows, indent="  ") -> list:
    out = [f"{indent}{'signal':<26}{'n':>6}{'rho':>8}{'dir':>5}"
           f"{'accuracy':>10}{'random':>9}{'lift':>8}"]
    for row in sorted(rows, key=lambda r: -(r["lift"] or -9)):
        if not row["available"]:
            out.append(f"{indent}{row['signal']:<26}{'--':>6}"
                       f"{'not present in this run':>40}")
            continue
        agrees = ("  ok" if row["direction_agrees"] else
                  " <-!" if row["direction_agrees"] is False else "   -")
        out.append(f"{indent}{row['signal']:<26}{row['available']:>6}"
                   f"{_fmt(row['rho'])}{agrees:>5}"
                   f"{_fmt(row['accuracy_at_coverage'], 10)}"
                   f"{_fmt(row['baseline'], 9)}"
                   f"{_fmt(row['lift'], 8)}")
    return out


def render(data: dict) -> str:
    out = ["", "SIGNALS  -  what predicts a bad extraction, other than the model", ""]
    out.append(f"  documents            {data['documents']:>8}")
    out.append(f"  routing coverage     {data['coverage']:>8.0%}   "
               f"(the least promising documents go to a person)")
    out.append("")
    out.append("  POOLED")
    out.extend(_table(data["pooled"]))
    out.append("")
    out.append("  rho is rank correlation with field accuracy. `dir` is whether the")
    out.append("  sign matches what was expected before looking; `<-!` did not.")
    out.append("  lift is accuracy over sending the same number of documents at random.")
    out.append("  A signal worth routing on has a lift, not merely a correlation.")

    for heading, key in (("BY DOCUMENT TYPE", "by_truth"),
                         ("BY DEGRADATION", "by_profile")):
        groups = data.get(key) or {}
        if len(groups) < 2:
            continue
        out.append("")
        out.append(f"  {heading}")
        for name, rows in groups.items():
            best = max(rows, key=lambda r: (r["lift"] or -9))
            if best["lift"] is None:
                out.append(f"    {name:<24} no signal available")
                continue
            out.append(f"    {name:<24} best: {best['signal']:<24}"
                       f"lift {best['lift']:+.3f}   "
                       f"({_fmt(best['accuracy_at_coverage'], 5).strip()} vs "
                       f"{_fmt(best['baseline'], 5).strip()})")
    return "\n".join(out)
