"""Scoring for the classifier stage.

Accuracy is the wrong headline here and it is worth being explicit about why.

The corpus is skewed -- 160 forms against 40 resumes -- so always answering `form`
scores 0.45 having read nothing. Any figure that cannot be compared against that is
unreadable, which is why the baseline classifier exists and why `majority_baseline` is
reported alongside every result.

Worse, one number cannot say *which* types get confused, and the confusions are not
equal. Calling a multi-bill invoice an invoice is a genuinely hard distinction: both
say "Invoice", and only the repeated per-service structure separates them. Calling a
resume an invoice means something is badly wrong. Both cost one point.

And a classification error is not contained. The predicted type selects the extraction
schema, so a misread document is then asked for fields it never had -- the same
structural trap as Phase 2's fabricated values, one stage earlier. Per-class recall is
what says how much extraction damage a classifier is doing, and to which type.

Abstentions are counted apart from errors, deliberately. A classifier that says
"unknown" on a hard document has done something useful: that document can be routed to
a person. One that guesses wrong has quietly corrupted the stage after it.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


def _rate(numerator: int, denominator: int):
    return round(numerator / denominator, 4) if denominator else None


@dataclass
class ClassificationScore:
    """Confusion between true and predicted types, and what falls out of it."""

    matrix: dict = field(default_factory=lambda: defaultdict(Counter))
    truths: Counter = field(default_factory=Counter)
    abstained: Counter = field(default_factory=Counter)
    runner_up_correct: int = 0
    seconds: float = 0.0

    def add(self, truth: str, predicted: str, runner_up: str = "",
            seconds: float = 0.0) -> None:
        self.truths[truth] += 1
        self.seconds += seconds
        if not predicted:
            self.abstained[truth] += 1
            return
        self.matrix[truth][predicted] += 1
        if predicted != truth and runner_up == truth:
            # Wrong, but it had the right answer second. That is a threshold problem,
            # not a comprehension one, and the two deserve different fixes.
            self.runner_up_correct += 1

    @property
    def total(self) -> int:
        return sum(self.truths.values())

    @property
    def correct(self) -> int:
        return sum(self.matrix[t][t] for t in self.truths)

    @property
    def answered(self) -> int:
        return self.total - sum(self.abstained.values())

    def majority_baseline(self):
        """What always guessing the commonest type would score, having read nothing."""
        return _rate(max(self.truths.values()), self.total) if self.truths else None

    def per_class(self) -> list:
        rows = []
        for truth in sorted(self.truths):
            support = self.truths[truth]
            hit = self.matrix[truth][truth]
            # Predicted as this type across every true type -- the denominator for
            # precision, which says how much of what it called X really was X.
            claimed = sum(self.matrix[other][truth] for other in self.truths)
            confusions = Counter({p: n for p, n in self.matrix[truth].items()
                                  if p != truth})
            worst, worst_n = (confusions.most_common(1) or [("", 0)])[0]
            rows.append({
                "type": truth,
                "support": support,
                "recall": _rate(hit, support),
                "precision": _rate(hit, claimed),
                "abstained": self.abstained[truth],
                "confused_with": worst or None,
                "confused_n": worst_n or None,
            })
        return rows

    def to_dict(self) -> dict:
        return {
            "documents": self.total,
            "answered": self.answered,
            "abstained": sum(self.abstained.values()),
            # Over everything, so abstentions count against it. A classifier cannot
            # buy accuracy by declining the hard half.
            "accuracy": _rate(self.correct, self.total),
            # Over what it actually answered: how right it is when it commits.
            "precision_answered": _rate(self.correct, self.answered),
            "majority_baseline": self.majority_baseline(),
            "right_answer_was_second": self.runner_up_correct or None,
            "seconds_per_document": _rate_float(self.seconds, self.total),
            "per_class": self.per_class(),
            "matrix": {t: dict(p) for t, p in sorted(self.matrix.items())},
        }


def _rate_float(total: float, n: int):
    return round(total / n, 2) if n else None


def render(score: ClassificationScore) -> str:
    data = score.to_dict()
    out = ["", "CLASSIFICATION", ""]
    out.append(f"  documents            {data['documents']:>8}")
    out.append(f"  accuracy             {_fmt(data['accuracy'])}")
    out.append(f"  majority baseline    {_fmt(data['majority_baseline'])}"
               "   <- always guessing the commonest type")
    if data["abstained"]:
        out.append(f"  abstained            {data['abstained']:>8}"
                   f"   (precision when it answered {_fmt(data['precision_answered']).strip()})")
    if data["right_answer_was_second"]:
        out.append(f"  right answer 2nd     {data['right_answer_was_second']:>8}"
                   "   <- a threshold problem, not a comprehension one")
    out.append("")
    out.append(f"  {'type':<22}{'n':>5}{'recall':>9}{'precision':>11}"
               f"{'abstained':>11}  most confused with")
    for row in data["per_class"]:
        confused = (f"{row['confused_with']} x{row['confused_n']}"
                    if row["confused_with"] else "")
        out.append(f"  {row['type']:<22}{row['support']:>5}{_fmt(row['recall'], 9)}"
                   f"{_fmt(row['precision'], 11)}{row['abstained']:>11}  {confused}")
    return "\n".join(out)


def _fmt(value, width: int = 9) -> str:
    return f"{'--' if value is None else f'{value:.3f}':>{width}}"
