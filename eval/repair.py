"""Whether repair helped, and the two ways that question is usually answered wrongly.

Phase 6 ends with a repair success rate. Almost every obvious definition of that number
is one a repair loop can satisfy without improving a single document, so the definition
is most of the work.

**The first trap: scoring repair by whether the complaints stopped.** A document enters
repair because a gate fired -- a validator found that subtotal plus tax does not equal
total, or too many fields came back blank. If success means the gate no longer fires,
the shortest path to success is to blank the fields the rule reads. An empty tax cannot
fail an arithmetic check. A loop optimised against its own critics learns to silence
them, and every dashboard would show it working.

So success is measured against the corpus labels, never against the gates. `gates_clear`
is still reported, because the difference between it and the real number is itself the
diagnostic: a repair that clears gates without moving field accuracy is doing exactly
the thing above.

**The second trap: reporting only the documents that improved.** Repair rewrites answers
that were partly right. Some come back worse, and a loop that improves sixty documents
and damages fifty is not a loop that improved sixty documents. `damaged` sits next to
`improved` in every table here and in the headline sentence, because reporting the
first without the second is how a net-negative change ships.

**The baseline: a blind re-run.** The extractor is sampled, so asking it the same
question twice changes some answers and improves some of them by luck. Any repair that
sends a second request inherits that for free. The question is whether *feedback* is
worth anything beyond a second roll of the dice, and it is answered by running an arm
that re-extracts with no feedback at all and scoring it identically. If guided repair
does not beat blind re-run, the guidance is decoration and the honest report says so.

Scores are per document and continuous -- a document is 24 fields of which 22 are
right -- for the same reason `eval.calibration` scores extraction that way. A pass/fail
bar here would hide small repairs and small damage equally.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# A change smaller than this is not a repair, it is the model answering the same
# question slightly differently. Set at one part in a thousand: field accuracy on the
# smallest schema here moves in steps of about 1/7, so anything at this scale is
# floating-point noise rather than a document changing.
EPSILON = 1e-3


def _rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


@dataclass
class Outcome:
    """One document through one repair arm."""

    file: str
    before: float                  # field accuracy as first extracted
    after: float                   # field accuracy after the arm ran
    gates_before: int              # how many gates fired before
    gates_after: int
    attempts: int = 0              # model calls this document cost
    doc_type: str = ""
    profile: str = "clean"
    error: str = ""                # the arm failed; the document keeps `before`

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def improved(self) -> bool:
        return self.delta > EPSILON

    @property
    def damaged(self) -> bool:
        return self.delta < -EPSILON

    @property
    def gates_clear(self) -> bool:
        return self.gates_after == 0 and self.gates_before > 0


@dataclass
class RepairScore:
    """One arm: what it cost, what it fixed, and what it broke."""

    arm: str = "repair"
    outcomes: list = field(default_factory=list)

    def add(self, outcome: Outcome) -> None:
        self.outcomes.append(outcome)

    def __len__(self) -> int:
        return len(self.outcomes)

    def to_dict(self) -> dict:
        rows = self.outcomes
        if not rows:
            return {"arm": self.arm, "documents": 0}
        improved = [r for r in rows if r.improved]
        damaged = [r for r in rows if r.damaged]
        before = sum(r.before for r in rows) / len(rows)
        after = sum(r.after for r in rows) / len(rows)
        return {
            "arm": self.arm,
            "documents": len(rows),
            "attempts": sum(r.attempts for r in rows),
            "errors": sum(1 for r in rows if r.error),

            "accuracy_before": round(before, 4),
            "accuracy_after": round(after, 4),
            # The headline. Net, because a loop is a single decision to run or not and
            # what it is worth is the sum of its help and its harm.
            "net_delta": round(after - before, 4),

            "improved": len(improved),
            "damaged": len(damaged),
            "unchanged": len(rows) - len(improved) - len(damaged),
            # Reported as a pair, always. "Repaired 60%" means nothing without the
            # share it broke, and quoting the first alone is how a net-negative loop
            # gets shipped.
            "improved_rate": _rate(len(improved), len(rows)),
            "damaged_rate": _rate(len(damaged), len(rows)),
            "gain_when_improved": (round(sum(r.delta for r in improved)
                                         / len(improved), 4) if improved else None),
            "loss_when_damaged": (round(sum(r.delta for r in damaged)
                                        / len(damaged), 4) if damaged else None),

            # Gate agreement is a diagnostic, never the score. A loop that clears gates
            # far more often than it improves documents is optimising against its
            # critics rather than against the page.
            "gates_clear": sum(1 for r in rows if r.gates_clear),
            "gates_clear_rate": _rate(sum(1 for r in rows if r.gates_clear), len(rows)),
            # Per document, because the arms saw the same documents and the only
            # honest comparison between them is paired. Without these the report can
            # say two arms differ by 0.011 and nothing about whether that survives
            # another forty documents -- which is exactly the question two runs of
            # this comparison have already disagreed about.
            "deltas": {r.file: round(r.delta, 4) for r in rows},
            # Per document alongside the delta, so "the gates went quiet while the
            # documents got worse" can be checked one document at a time instead of
            # inferred from two totals. On degraded documents that pattern is the
            # finding, and a claim that large should not rest on arithmetic done in
            # a commit message.
            "gates": {r.file: [r.gates_before, r.gates_after] for r in rows},
        }


def paired(a: dict, b: dict) -> dict:
    """Arm `a` against arm `b`, document by document.

    Paired, because both arms ran over the identical document set. The unpaired
    difference of two means throws away the strongest thing known about this
    comparison -- that a document hard for one arm was hard for the other -- and
    unpaired noise on forty documents is far larger than the effect being looked for.

    The interval is the ordinary normal approximation on the mean of the per-document
    differences. It is reported because the alternative is quoting a difference with no
    sense of its width, and two runs of this comparison have already come out with
    opposite signs -- which is not a contradiction, it is what an effect smaller than
    its own error bar looks like when you run it twice.
    """
    shared = sorted(set(a.get("deltas") or {}) & set(b.get("deltas") or {}))
    if len(shared) < 3:
        return {"documents": len(shared), "mean": None, "stderr": None,
                "interval": None, "resolvable": False}
    diffs = [a["deltas"][key] - b["deltas"][key] for key in shared]
    n = len(diffs)
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    stderr = (variance / n) ** 0.5
    half = 1.96 * stderr
    return {
        "documents": n,
        "mean": round(mean, 4),
        "stderr": round(stderr, 4),
        "interval": [round(mean - half, 4), round(mean + half, 4)],
        # Whether the interval excludes zero. When it does not, the difference has not
        # been measured -- it has been observed once, and the honest report says so
        # rather than quoting the point estimate.
        "resolvable": (mean - half > 0) or (mean + half < 0),
        "better": sum(1 for d in diffs if d > EPSILON),
        "worse": sum(1 for d in diffs if d < -EPSILON),
        "tied": sum(1 for d in diffs if abs(d) <= EPSILON),
    }


def compare(arms: dict) -> dict:
    """Several arms scored together, each against the blind re-run.

    `arms` is {name: RepairScore}. The arm named `rerun` is the baseline if present:
    the extractor asked the same question again with no feedback. Everything else is
    reported as a delta from it, because that is the only comparison that isolates what
    the feedback was worth from what a second sample was worth.
    """
    scored = {name: score.to_dict() for name, score in arms.items()}
    baseline = scored.get("rerun")
    for name, row in scored.items():
        if not baseline or name == "rerun" or not row.get("documents"):
            row["over_rerun"] = None
            continue
        row["over_rerun"] = round((row["net_delta"] or 0)
                                  - (baseline["net_delta"] or 0), 4)
    pairs = {}
    if baseline:
        for name, row in scored.items():
            if name != "rerun" and row.get("documents"):
                pairs[name] = paired(row, baseline)
    return {"arms": scored,
            "baseline": "rerun" if baseline else None,
            "paired": pairs,
            "documents": max((r.get("documents", 0) for r in scored.values()),
                             default=0)}


def goodhart(arm: dict) -> dict:
    """Did the gates go quiet on the same documents that got worse?

    The aggregate version of this question -- 52 gates cleared, 52 documents damaged --
    is suggestive and proves nothing. Two disjoint groups of 52 would produce identical
    totals and mean something entirely different: rules being satisfied on one set of
    documents while a separate set degraded. Only the per-document join separates
    "silencing the critics" from "two unrelated things happened".

    The cell that matters is `cleared_and_damaged`: a document that stopped tripping
    every rule *and* came back further from the truth. There is no benign reading of
    that. It is what blanking the fields a rule reads produces, and it is the reason
    success in this module is defined against the corpus and never against the gates.

    `lift` is how much more likely a damaged document was to clear its gates than an
    undamaged one. Above 1.0 means the rules are being satisfied by exactly the changes
    that hurt.
    """
    deltas = arm.get("deltas") or {}
    gates = arm.get("gates") or {}
    shared = sorted(set(deltas) & set(gates))
    if not shared:
        return {"documents": 0, "available": False}

    cells = {"cleared_and_damaged": 0, "cleared_and_improved": 0,
             "cleared_and_unchanged": 0, "held_and_damaged": 0,
             "held_and_improved": 0, "held_and_unchanged": 0}
    cleared_damaged = cleared_ok = damaged = intact = 0
    for key in shared:
        delta = deltas[key]
        before, after = gates[key]
        # Only a document that had something to clear can clear it.
        cleared = before > 0 and after == 0
        state = ("damaged" if delta < -EPSILON
                 else "improved" if delta > EPSILON else "unchanged")
        cells[f"{'cleared' if cleared else 'held'}_and_{state}"] += 1
        if state == "damaged":
            damaged += 1
            cleared_damaged += cleared
        else:
            intact += 1
            cleared_ok += cleared

    rate_damaged = cleared_damaged / damaged if damaged else None
    rate_intact = cleared_ok / intact if intact else None
    return {
        "documents": len(shared),
        "available": True,
        **cells,
        "clear_rate_when_damaged": (round(rate_damaged, 4)
                                    if rate_damaged is not None else None),
        "clear_rate_otherwise": (round(rate_intact, 4)
                                 if rate_intact is not None else None),
        "lift": (round(rate_damaged / rate_intact, 3)
                 if rate_damaged and rate_intact else None),
    }


def by_slice(score: RepairScore, key: str = "doc_type") -> list:
    """Per type or per profile. Repair is not uniform and an average hides which half."""
    groups = defaultdict(list)
    for row in score.outcomes:
        groups[getattr(row, key, "") or "unknown"].append(row)
    out = []
    for name, rows in sorted(groups.items()):
        improved = sum(1 for r in rows if r.improved)
        damaged = sum(1 for r in rows if r.damaged)
        out.append({
            key: name,
            "documents": len(rows),
            "net_delta": round(sum(r.delta for r in rows) / len(rows), 4),
            "improved": improved,
            "damaged": damaged,
        })
    return out


def _needed(pair) -> str:
    """Roughly how many documents would put the observed effect outside the interval.

    Standard error falls as 1/sqrt(n), so the multiplier is (1.96 * sd / effect)^2.
    Deliberately rough and rendered as an order of magnitude: the point is to say
    whether the answer is another forty documents or another four thousand, which is a
    decision about whether the question is worth pursuing at all.
    """
    if not pair.get("mean") or not pair.get("stderr"):
        return "many more"
    n = pair["documents"]
    sd = pair["stderr"] * (n ** 0.5)
    needed = ((1.96 * sd) / abs(pair["mean"])) ** 2
    if needed > 100000:
        return "far more than this corpus holds"
    return f"~{int(needed):,}"


def _fmt(value, width=8, digits=3, sign=False):
    if value is None:
        return f"{'--':>{width}}"
    return f"{value:>{'+' if sign else ''}{width}.{digits}f}"


def render(data: dict, slices: dict = None) -> str:
    out = ["", "REPAIR  -  did the second attempt help", ""]
    out.append(f"  documents            {data.get('documents', 0):>8}")
    if data.get("baseline"):
        out.append("  baseline             blind re-run, same prompt, no feedback")
    else:
        out.append("  baseline                 none   <- no blind re-run arm was run;")
        out.append("     any gain below includes whatever a second sample is worth")
    out.append("")
    out.append(f"  {'arm':<14}{'calls':>7}{'before':>9}{'after':>9}{'net':>9}"
               f"{'better':>8}{'worse':>7}{'gates':>7}{'vs rerun':>10}")
    for name, row in data["arms"].items():
        if not row.get("documents"):
            out.append(f"  {name:<14}{'no documents':>50}")
            continue
        out.append(
            f"  {name:<14}{row['attempts']:>7}"
            f"{_fmt(row['accuracy_before'], 9)}{_fmt(row['accuracy_after'], 9)}"
            f"{_fmt(row['net_delta'], 9, sign=True)}"
            f"{row['improved']:>8}{row['damaged']:>7}{row['gates_clear']:>7}"
            f"{_fmt(row.get('over_rerun'), 10, sign=True)}")
    harmful = [name for name, row in data["arms"].items()
               if row.get("documents") and (row.get("net_delta") or 0) < 0]
    if harmful:
        out.append("")
        out.append(f"  NET-NEGATIVE: {', '.join(harmful)} left documents worse than "
                   f"they were found.")
        out.append("  Repair is optional. An arm that scores below zero should be off.")
    out.append("")
    out.append("  `better` and `worse` are documents, scored against the corpus labels.")
    out.append("  `gates` is how many stopped tripping a rule -- a diagnostic, not the")
    out.append("  score. A loop clearing far more gates than it improves documents is")
    out.append("  silencing its critics, which it can do by blanking the fields they read.")

    for name, pair in (data.get("paired") or {}).items():
        out.append("")
        if pair["mean"] is None:
            out.append(f"  {name} against rerun: too few shared documents to compare.")
            continue
        low, high = pair["interval"]
        out.append(f"  {name} against rerun, paired over {pair['documents']} documents:")
        out.append(f"    mean difference {pair['mean']:+.4f}  "
                   f"95% interval [{low:+.4f}, {high:+.4f}]")
        out.append(f"    better on {pair['better']}, worse on {pair['worse']}, "
                   f"tied on {pair['tied']}")
        if not pair["resolvable"]:
            out.append("    The interval does not exclude zero. This difference has")
            out.append("    not been measured, only observed once -- do not quote the")
            out.append(f"    point estimate. Resolving it needs roughly "
                       f"{_needed(pair)} documents.")
        elif pair["mean"] > 0:
            out.append("    The feedback is worth something beyond a second sample.")
        else:
            out.append("    The feedback is worse than a second sample alone.")
        # A resolved win between two arms says nothing about whether either should
        # run. Both can be harmful, and then "beats the baseline" means "does less
        # damage" -- which is the single most quotable-out-of-context line this
        # report can produce, so the qualification is attached to it rather than
        # left further down the page.
        arm_row = data["arms"].get(name) or {}
        base_row = data["arms"].get("rerun") or {}
        if (arm_row.get("net_delta") or 0) < 0 and (base_row.get("net_delta") or 0) < 0:
            out.append("    But BOTH arms are net-negative here. This is less harmful,")
            out.append("    not helpful. Neither should run on these documents.")

    for name, row in data["arms"].items():
        if row.get("documents") and row["damaged"] > row["improved"]:
            out.append("")
            out.append(f"  {name} damaged more documents than it improved "
                       f"({row['damaged']} against {row['improved']}).")

    for name, row in data["arms"].items():
        table = goodhart(row)
        if not table.get("available"):
            continue
        cad = table["cleared_and_damaged"]
        if not cad and not table["cleared_and_improved"]:
            continue
        out.append("")
        out.append(f"  GATES AGAINST TRUTH  -  {name}")
        out.append(f"  {'':<22}{'damaged':>10}{'improved':>10}{'unchanged':>11}")
        out.append(f"  {'gates cleared':<22}{cad:>10}"
                   f"{table['cleared_and_improved']:>10}"
                   f"{table['cleared_and_unchanged']:>11}")
        out.append(f"  {'gates still firing':<22}{table['held_and_damaged']:>10}"
                   f"{table['held_and_improved']:>10}"
                   f"{table['held_and_unchanged']:>11}")
        if cad:
            out.append(f"  {cad} documents stopped tripping every rule AND came back "
                       f"further from the truth.")
            out.append("  There is no benign reading of that cell.")
        if table.get("lift") and table["lift"] > 1.0:
            out.append(f"  A damaged document was {table['lift']:.2f}x more likely to "
                       f"clear its gates than an undamaged one:")
            out.append(f"  {table['clear_rate_when_damaged']:.1%} against "
                       f"{table['clear_rate_otherwise']:.1%}. The rules are being "
                       f"satisfied by the changes that hurt.")

    for key, rows in (slices or {}).items():
        out.append("")
        out.append(f"  BY {key.replace('_', ' ').upper()}")
        out.append(f"  {key:<24}{'n':>6}{'net':>9}{'better':>8}{'worse':>7}")
        for row in rows:
            out.append(f"  {str(row[key])[:23]:<24}{row['documents']:>6}"
                       f"{_fmt(row['net_delta'], 9, sign=True)}"
                       f"{row['improved']:>8}{row['damaged']:>7}")
    return "\n".join(out)
