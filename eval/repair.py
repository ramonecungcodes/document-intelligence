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
    return {"arms": scored,
            "baseline": "rerun" if baseline else None,
            "documents": max((r.get("documents", 0) for r in scored.values()),
                             default=0)}


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
    out.append("")
    out.append("  `better` and `worse` are documents, scored against the corpus labels.")
    out.append("  `gates` is how many stopped tripping a rule -- a diagnostic, not the")
    out.append("  score. A loop clearing far more gates than it improves documents is")
    out.append("  silencing its critics, which it can do by blanking the fields they read.")

    for name, row in data["arms"].items():
        if not row.get("documents"):
            continue
        verdict = row.get("over_rerun")
        if verdict is not None and verdict <= 0:
            out.append("")
            out.append(f"  {name} does not beat a blind re-run "
                       f"({verdict:+.4f}). The feedback is worth nothing here;")
            out.append("  the gain is what a second sample was worth.")
        if row["damaged"] > row["improved"]:
            out.append("")
            out.append(f"  {name} damaged more documents than it improved "
                       f"({row['damaged']} against {row['improved']}).")

    for key, rows in (slices or {}).items():
        out.append("")
        out.append(f"  BY {key.replace('_', ' ').upper()}")
        out.append(f"  {key:<24}{'n':>6}{'net':>9}{'better':>8}{'worse':>7}")
        for row in rows:
            out.append(f"  {str(row[key])[:23]:<24}{row['documents']:>6}"
                       f"{_fmt(row['net_delta'], 9, sign=True)}"
                       f"{row['improved']:>8}{row['damaged']:>7}")
    return "\n".join(out)
