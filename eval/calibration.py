"""Whether the pipeline's confidence means anything, and where to put the floor.

Every earlier phase asked how often the system is right. This one asks a different
question: when it is wrong, did it know? A model at 0.99 that is wrong a third of the
time is worse than a model at 0.70 that is wrong a third of the time, because the first
one cannot be routed around and the second one can.

Three numbers, and the third is the only one anybody acts on.

**Reliability.** Bin the documents by confidence and compare each bin's confidence to
its accuracy. A calibrated model's 0.8 bin is right 80% of the time. ECE is the
document-weighted mean of those gaps, and MCE the worst single bin -- reported apart
because a model can be excellent on average and untrustworthy in exactly the band a
threshold would sit in, which is the band with the fewest documents and so the least
weight in the mean.

**Separation.** Whether confidence orders the errors below the correct answers at all.
This is where the baseline lives, and it is not a courtesy: abstaining on the least
confident 20% of documents is only worth doing if it beats abstaining on a *random*
20%. Random abstention leaves accuracy exactly where it started, so that baseline is a
flat line at the overall accuracy, and it needs no sampling to compute -- the expected
accuracy of a uniformly random subset is the accuracy of the whole. Every point where
the real curve sits on that line is a threshold buying nothing.

**Coverage and accuracy.** For each threshold: how many documents get answered, how
accurate those answers are, and how many errors escape. The floor is chosen from this
table and nowhere else, by naming a tolerable error rate and reading off the coverage
that comes with it.

Abstained documents are included, which is the whole reason `withheld` is recorded.
Scoring only the answered ones measures a policy against the documents it already
accepted, and reports every floor as free.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

BINS = 10


def _rate(numerator: int, denominator: int):
    return round(numerator / denominator, 4) if denominator else None


@dataclass
class Observation:
    """One decision, and whether it was right.

    `confidence` is what the classifier said; `correct` compares what it was going to
    answer -- withheld or not -- against the truth. Abstention is a policy applied on
    top of these, not a property of them, which is what lets one set of observations
    answer questions about every threshold rather than only the one that happened to be
    in force during the run.
    """
    confidence: float
    correct: bool
    profile: str = "clean"
    truth: str = ""
    answer: str = ""


@dataclass
class CalibrationScore:
    """Confidence against correctness, over a set of decisions."""

    observations: list = field(default_factory=list)
    # Decisions with no confidence at all. A keyword classifier reports none, and
    # averaging over the ones that do while silently dropping the rest describes a
    # different pipeline from the one that ran.
    unscored: int = 0

    def add(self, confidence, correct: bool, profile: str = "clean",
            truth: str = "", answer: str = "") -> None:
        if confidence is None:
            self.unscored += 1
            return
        self.observations.append(
            Observation(float(confidence), bool(correct), profile or "clean",
                        truth, answer))

    def __len__(self) -> int:
        return len(self.observations)

    # ---------------------------------------------------------------- reliability

    def bins(self, rows=None) -> list:
        """Fixed-width confidence bins, empty ones included.

        Empty bins are kept because their absence misleads in the wrong direction: a
        table that jumps from 0.5 to 0.9 looks like a smooth curve with a gap, when what
        it means is that the model never once expressed middling confidence -- a fact
        about the model worth seeing.
        """
        rows = self.observations if rows is None else rows
        buckets = [[] for _ in range(BINS)]
        for row in rows:
            index = min(int(row.confidence * BINS), BINS - 1)
            buckets[index].append(row)
        out = []
        for index, bucket in enumerate(buckets):
            correct = sum(1 for r in bucket if r.correct)
            out.append({
                "low": round(index / BINS, 2),
                "high": round((index + 1) / BINS, 2),
                "n": len(bucket),
                "mean_confidence": (round(sum(r.confidence for r in bucket)
                                          / len(bucket), 4) if bucket else None),
                "accuracy": _rate(correct, len(bucket)),
            })
        return out

    def expected_calibration_error(self, rows=None):
        """Document-weighted mean gap between stated confidence and observed accuracy."""
        rows = self.observations if rows is None else rows
        if not rows:
            return None
        total = 0.0
        for row in self.bins(rows):
            if row["n"]:
                total += (row["n"] / len(rows)) * abs(row["accuracy"]
                                                      - row["mean_confidence"])
        return round(total, 4)

    def max_calibration_error(self, rows=None):
        """The worst bin holding a meaningful number of documents.

        Bins under five are excluded. A single document in the 0.3 bin produces an
        accuracy of exactly 0 or exactly 1 and a gap that is an artifact of the bin
        width; letting that be the headline turns a real problem into an unbelievable
        one.
        """
        rows = self.observations if rows is None else rows
        gaps = [abs(b["accuracy"] - b["mean_confidence"])
                for b in self.bins(rows) if b["n"] >= 5]
        return round(max(gaps), 4) if gaps else None

    def brier(self, rows=None):
        """Mean squared error of the confidence, read as P(correct).

        One number covering both calibration and separation, which is why it sits
        alongside ECE rather than replacing it: a model can improve its Brier score by
        getting more answers right without its confidence becoming any more honest.
        """
        rows = self.observations if rows is None else rows
        if not rows:
            return None
        return round(sum((r.confidence - r.correct) ** 2 for r in rows) / len(rows), 4)

    # ------------------------------------------------------------ coverage curve

    def curve(self, rows=None, thresholds=None) -> list:
        """Answer at or above the threshold, decline below it, and score what was
        answered.

        `baseline_accuracy` on each row is what declining the same *number* of documents
        at random would have produced, which is the overall accuracy unchanged. The
        comparison is the point: a threshold is worth its abstentions only by the
        distance between those two columns.
        """
        rows = self.observations if rows is None else rows
        if not rows:
            return []
        base = _rate(sum(1 for r in rows if r.correct), len(rows))
        if thresholds is None:
            thresholds = [i / 20 for i in range(21)]
        out = []
        for threshold in thresholds:
            answered = [r for r in rows if r.confidence >= threshold]
            wrong = sum(1 for r in answered if not r.correct)
            # Errors caught: wrong answers that fell below the floor. That is what the
            # abstentions bought, and it belongs beside what they cost.
            caught = sum(1 for r in rows
                         if r.confidence < threshold and not r.correct)
            out.append({
                "threshold": round(threshold, 4),
                "answered": len(answered),
                "coverage": _rate(len(answered), len(rows)),
                "accuracy": _rate(len(answered) - wrong, len(answered)),
                "errors": wrong,
                "errors_caught": caught,
                "baseline_accuracy": base,
            })
        return out

    def operating_point(self, target_accuracy: float = 0.99, rows=None):
        """The most permissive floor that still holds accuracy at the target.

        Most permissive, not lowest-error: the question a floor answers is how much of
        the corpus can be automated at a stated error rate, so among the thresholds that
        clear the bar the useful one is the one declining fewest documents. Returns None
        when none of them reach the target, which is a real answer -- it means this
        model cannot be routed to that standard at any coverage.
        """
        best = None
        for row in self.curve(rows):
            if row["answered"] and row["accuracy"] >= target_accuracy:
                if best is None or row["coverage"] > best["coverage"]:
                    best = row
        return best

    # ------------------------------------------------------------------ slicing

    def by_profile(self) -> dict:
        groups = defaultdict(list)
        for row in self.observations:
            groups[row.profile].append(row)
        return dict(groups)

    def to_dict(self, target_accuracy: float = 0.99) -> dict:
        rows = self.observations
        correct = sum(1 for r in rows if r.correct)
        out = {
            "documents": len(rows),
            "unscored": self.unscored,
            "accuracy": _rate(correct, len(rows)),
            "mean_confidence": (round(sum(r.confidence for r in rows) / len(rows), 4)
                                if rows else None),
            "ece": self.expected_calibration_error(),
            "mce": self.max_calibration_error(),
            "brier": self.brier(),
            "bins": self.bins(),
            "curve": self.curve(),
            "target_accuracy": target_accuracy,
            "operating_point": self.operating_point(target_accuracy),
            "profiles": [],
        }
        # Overconfidence is the direction that matters, and only the signed gap shows
        # it: a model 0.1 under and a model 0.1 over have the same ECE and opposite
        # consequences, because only one of them routes its errors past a floor.
        out["mean_gap"] = (round(out["mean_confidence"] - out["accuracy"], 4)
                           if rows else None)
        for profile, group in sorted(self.by_profile().items()):
            hits = sum(1 for r in group if r.correct)
            out["profiles"].append({
                "profile": profile,
                "documents": len(group),
                "accuracy": _rate(hits, len(group)),
                "mean_confidence": round(sum(r.confidence for r in group)
                                         / len(group), 4),
                "ece": self.expected_calibration_error(group),
                "brier": self.brier(group),
                "operating_point": self.operating_point(target_accuracy, group),
            })
        return out


def _fmt(value, width=8):
    return f"{'--':>{width}}" if value is None else f"{value:>{width}.3f}"


def render(score: CalibrationScore, target_accuracy: float = 0.99) -> str:
    d = score.to_dict(target_accuracy)
    out = ["", "CALIBRATION  -  is the confidence real", ""]
    out.append(f"  decisions            {d['documents']:>8}")
    if d["unscored"]:
        out.append(f"  no confidence given  {d['unscored']:>8}   (not scored below)")
    if not d["documents"]:
        out.append("")
        out.append("  nothing to score.")
        return "\n".join(out)
    out.append(f"  accuracy            {_fmt(d['accuracy'])}")
    out.append(f"  mean confidence     {_fmt(d['mean_confidence'])}")
    gap = d["mean_gap"]
    if gap is not None:
        way = "overconfident" if gap > 0 else "underconfident"
        out.append(f"  gap                 {gap:>+8.3f}   {way}")
    out.append(f"  ECE                 {_fmt(d['ece'])}   mean |confidence - accuracy|")
    out.append(f"  MCE                 {_fmt(d['mce'])}   worst bin of 5 or more")
    out.append(f"  Brier               {_fmt(d['brier'])}   lower is better")

    out.append("")
    out.append("  RELIABILITY")
    out.append(f"  {'bin':<14}{'n':>6}{'confidence':>12}{'accuracy':>11}{'gap':>9}")
    for row in d["bins"]:
        label = f"{row['low']:.1f} - {row['high']:.1f}"
        if not row["n"]:
            out.append(f"  {label:<14}{0:>6}{'--':>12}{'--':>11}{'--':>9}")
            continue
        out.append(f"  {label:<14}{row['n']:>6}{row['mean_confidence']:>12.3f}"
                   f"{row['accuracy']:>11.3f}"
                   f"{row['accuracy'] - row['mean_confidence']:>+9.3f}")

    out.append("")
    out.append("  COVERAGE  -  answer at or above the floor, decline below it")
    out.append(f"  {'floor':<9}{'answered':>9}{'coverage':>10}{'accuracy':>10}"
               f"{'random':>9}{'errors':>8}{'caught':>8}")
    seen = set()
    for row in d["curve"]:
        # One row per distinct coverage. Twenty-one thresholds over a model that only
        # ever speaks from the top bin produce twenty identical lines, and a table that
        # long reads as more measurement than was taken.
        if row["coverage"] in seen:
            continue
        seen.add(row["coverage"])
        out.append(f"  {row['threshold']:<9.2f}{row['answered']:>9}"
                   f"{_fmt(row['coverage'], 10)}{_fmt(row['accuracy'], 10)}"
                   f"{_fmt(row['baseline_accuracy'], 9)}"
                   f"{row['errors']:>8}{row['errors_caught']:>8}")
    out.append("  random = declining the same number of documents at random, which")
    out.append("  leaves accuracy where it started. A floor is worth only the distance")
    out.append("  between those two columns.")

    point = d["operating_point"]
    out.append("")
    if point:
        out.append(f"  At {d['target_accuracy']:.0%} accuracy: floor "
                   f"{point['threshold']:.2f}, covering {point['coverage']:.1%} of "
                   f"documents ({point['answered']}), {point['errors']} errors "
                   f"through, {point['errors_caught']} declined.")
    else:
        out.append(f"  No floor reaches {d['target_accuracy']:.0%} accuracy. This model "
                   f"cannot be routed to that standard;")
        out.append("  its errors are not underneath its correct answers.")

    if len(d["profiles"]) > 1:
        out.append("")
        out.append("  BY DEGRADATION")
        out.append(f"  {'profile':<16}{'n':>6}{'accuracy':>10}{'confidence':>12}"
                   f"{'ECE':>8}{'floor':>8}{'coverage':>10}")
        for row in d["profiles"]:
            point = row["operating_point"]
            floor = f"{point['threshold']:>8.2f}" if point else f"{'none':>8}"
            cover = _fmt(point["coverage"], 10) if point else f"{'--':>10}"
            out.append(f"  {row['profile']:<16}{row['documents']:>6}"
                       f"{_fmt(row['accuracy'], 10)}{_fmt(row['mean_confidence'], 12)}"
                       f"{_fmt(row['ece'], 8)}{floor}{cover}")
        out.append(f"  floor and coverage are per profile, at "
                   f"{d['target_accuracy']:.0%}")
    return "\n".join(out)
