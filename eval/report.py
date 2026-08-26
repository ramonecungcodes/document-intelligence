"""The score report: a structured result, serialisable to JSON.

Scoring is kept separate from rendering so the same report can be printed as a table
now, drawn as charts by the `/eval` screen later, diffed by CI to catch regressions,
and compared across configurations for the extractor ablation.

Provenance and cost carry slots that nothing populates yet. They are here from the
first version on purpose: a report that cannot say which corpus, which model and which
knowledge pack produced it is unattributable, and adding those fields later would make
every report written before then worthless for comparison.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

REPORT_VERSION = "1"


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


@dataclass
class FieldScore:
    """One field, accumulated over the documents in a slice."""

    name: str
    kind: str = "text"
    n: int = 0            # graded occurrences
    exact: int = 0        # byte-equal
    match: int = 0        # equal once normalised
    blank: int = 0        # truth was empty and the prediction agreed
    missing: int = 0      # truth had a value, prediction did not
    spurious: int = 0     # prediction had a value, truth did not
    notes: Counter = field(default_factory=Counter)

    def add(self, comparison) -> None:
        self.n += 1
        self.exact += comparison.exact
        self.match += comparison.match
        if comparison.note == "both blank":
            self.blank += 1
        if comparison.note == "missing":
            self.missing += 1
        elif comparison.note.startswith("predicted a value"):
            self.spurious += 1
        elif comparison.normalised_only and comparison.note:
            self.notes[comparison.note.split()[0]] += 1

    def to_dict(self):
        return {
            "field": self.name,
            "kind": self.kind,
            "n": self.n,
            "exact": _rate(self.exact, self.n),
            "accuracy": _rate(self.match, self.n),
            # Fields the corpus deliberately empties are trivially "correct" when the
            # prediction is also empty, which flatters any extractor on the defective
            # set. Accuracy over the fields that actually carry a value is the number
            # to quote.
            "accuracy_nonblank": _rate(self.match - self.blank, self.n - self.blank),
            "blank": self.blank,
            "recovered_by_normalisation": self.match - self.exact,
            "missing": self.missing,
            "spurious": self.spurious,
            "notes": dict(self.notes.most_common(3)),
        }


@dataclass
class GroupScore:
    """A repeating group: rows are matched first, then fields within matched rows.

    Row recall is reported on its own because missing an entire row -- a whole
    billable service on a multi-bill invoice -- is a different and worse failure than
    getting one field wrong inside a row that was found.
    """

    name: str
    truth_rows: int = 0
    predicted_rows: int = 0
    matched_rows: int = 0
    fields: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "group": self.name,
            "truth_rows": self.truth_rows,
            "predicted_rows": self.predicted_rows,
            "matched_rows": self.matched_rows,
            "row_precision": _rate(self.matched_rows, self.predicted_rows),
            "row_recall": _rate(self.matched_rows, self.truth_rows),
            "row_f1": _rate(
                2 * self.matched_rows, self.predicted_rows + self.truth_rows
            ),
            "fields": [f.to_dict() for f in self.fields.values()],
            "groups": [g.to_dict() for g in self.groups.values()],
        }


@dataclass
class SliceScore:
    """Documents grouped by something worth comparing: type, layout, degradation."""

    name: str
    dimension: str = "doc_type"
    docs: int = 0
    scored: int = 0        # documents graded
    failed: int = 0        # extraction errored; not graded, counted here instead
    fields: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)

    def totals(self):
        n = sum(f.n for f in self.fields.values())
        match = sum(f.match for f in self.fields.values())
        exact = sum(f.exact for f in self.fields.values())
        blank = sum(f.blank for f in self.fields.values())
        return n, match, exact, blank

    def to_dict(self):
        n, match, exact, blank = self.totals()
        return {
            "slice": self.name,
            "dimension": self.dimension,
            "documents": self.docs,
            "scored": self.scored,
            "failed": self.failed,
            "field_accuracy": _rate(match, n),
            "field_accuracy_nonblank": _rate(match - blank, n - blank),
            "field_exact": _rate(exact, n),
            "blank_fields": blank,
            "fields": [f.to_dict() for f in self.fields.values()],
            "groups": [g.to_dict() for g in self.groups.values()],
        }


@dataclass
class DetectionScore:
    """Defect detection, scored against the `irregularities` ground truth.

    The false-positive rate on the clean corpus is the number that matters most: a
    validator suite that fires on clean documents floods the review queue, and only
    the clean set can reveal it.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    clean_docs: int = 0
    clean_docs_flagged: int = 0
    per_tag: dict = field(default_factory=dict)

    def add(self, tag: str, outcome: str) -> None:
        counts = self.per_tag.setdefault(tag, {"tp": 0, "fp": 0, "fn": 0})
        counts[outcome] += 1
        setattr(self, outcome, getattr(self, outcome) + 1)

    def to_dict(self):
        return {
            "precision": _rate(self.tp, self.tp + self.fp),
            "recall": _rate(self.tp, self.tp + self.fn),
            "f1": _rate(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "clean_documents": self.clean_docs,
            "clean_documents_flagged": self.clean_docs_flagged,
            "false_positive_rate_on_clean": _rate(self.clean_docs_flagged, self.clean_docs),
            "per_tag": {
                tag: {
                    **counts,
                    "recall": _rate(counts["tp"], counts["tp"] + counts["fn"]),
                }
                for tag, counts in sorted(self.per_tag.items())
            },
        }


@dataclass
class ScoreReport:
    provenance: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    slices: list = field(default_factory=list)
    detection: Optional[DetectionScore] = None
    warnings: list = field(default_factory=list)

    def overall(self):
        by_type = [s for s in self.slices if s.dimension == "doc_type"]
        totals = [s.totals() for s in by_type]
        n = sum(t[0] for t in totals)
        match = sum(t[1] for t in totals)
        exact = sum(t[2] for t in totals)
        blank = sum(t[3] for t in totals)
        return {
            "documents": sum(s.docs for s in by_type),
            "scored": sum(s.scored for s in by_type),
            "failed": sum(s.failed for s in by_type),
            "fields_graded": n,
            "field_accuracy": _rate(match, n),
            "field_accuracy_nonblank": _rate(match - blank, n - blank),
            "field_exact": _rate(exact, n),
            "blank_fields": blank,
        }

    def to_dict(self):
        return {
            "report_version": REPORT_VERSION,
            "provenance": self.provenance,
            "cost": self.cost,
            "overall": self.overall(),
            "slices": [s.to_dict() for s in self.slices],
            "detection": self.detection.to_dict() if self.detection else None,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False) + "\n"

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.to_json())
