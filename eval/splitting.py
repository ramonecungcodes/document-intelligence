"""Scoring for the splitter stage.

Boundary F1 is the headline, and it needs both halves reported beside it because the
two failures cost different things.

A missed boundary merges two documents. The classifier then types a chimera and the
extractor reads fields off pages belonging to two different documents -- one wrong
answer, and a confusing one, because the output looks like a document that never
existed.

A spurious boundary splits one document in half. Both halves get classified and
extracted, most fields come back absent from one half or the other, and the damage is
visible as missing data rather than as invented data. Worse for throughput, better for
trust.

So neither precision nor recall alone says whether a splitter is usable, and the
baselines exist to keep the F1 readable: `every_page` has recall 1.000 by construction
and `single` has precision undefined for the same reason. A splitter that beats neither
has not earned its latency.

`exact` is reported separately and is the number an operator actually feels: the share
of files that came out perfectly segmented, no merges and no cuts. A corpus of
three-document bundles can post a respectable F1 while getting almost no whole file
right, and that is the difference between a demo and a pipeline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


def _rate(numerator: int, denominator: int):
    return round(numerator / denominator, 4) if denominator else None


@dataclass
class SplitScore:
    """Boundary agreement between what was found and what was there."""

    found: int = 0
    truth: int = 0
    hit: int = 0
    files: int = 0
    exact: int = 0
    merged: int = 0          # boundaries that were there and were not found
    spurious: int = 0        # boundaries found that were not there
    same_type_truth: int = 0
    same_type_hit: int = 0
    seconds: float = 0.0

    def add(self, predicted, actual, doc_types=None, seconds: float = 0.0) -> None:
        p, a = set(predicted), set(actual)
        self.found += len(p)
        self.truth += len(a)
        self.hit += len(p & a)
        self.merged += len(a - p)
        self.spurious += len(p - a)
        self.files += 1
        self.exact += (p == a)
        self.seconds += seconds
        # The joins where the type does not change are the ones a per-page classifier
        # is blind to. Scoring them apart is what stops a splitter from looking solved
        # because the easy joins outnumber the hard ones.
        for index, boundary in enumerate(sorted(a)):
            if doc_types and index + 1 < len(doc_types) and \
                    doc_types[index] == doc_types[index + 1]:
                self.same_type_truth += 1
                self.same_type_hit += boundary in p

    @property
    def precision(self):
        return _rate(self.hit, self.found)

    @property
    def recall(self):
        return _rate(self.hit, self.truth)

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if p and r else None

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "boundaries_truth": self.truth,
            "boundaries_found": self.found,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact_files": _rate(self.exact, self.files),
            "merged": self.merged,
            "spurious": self.spurious,
            "same_type_recall": _rate(self.same_type_hit, self.same_type_truth),
            "same_type_boundaries": self.same_type_truth,
            "seconds_per_file": round(self.seconds / self.files, 2) if self.files else None,
        }


def render(score: SplitScore, label: str = "") -> str:
    d = score.to_dict()

    def fmt(v):
        return "     --" if v is None else f"{v:>7.3f}"

    out = ["", f"SPLITTING{(' - ' + label) if label else ''}", ""]
    out.append(f"  files                {d['files']:>7}")
    out.append(f"  boundaries           {d['boundaries_truth']:>7}   "
               f"found {d['boundaries_found']}")
    out.append(f"  precision           {fmt(d['precision'])}")
    out.append(f"  recall              {fmt(d['recall'])}")
    out.append(f"  F1                  {fmt(d['f1'])}")
    out.append(f"  files exactly right {fmt(d['exact_files'])}   "
               "<- no merges and no cuts, the number an operator feels")
    out.append("")
    out.append(f"  merged (missed)      {d['merged']:>7}   "
               "<- two documents become one; the extractor reads a chimera")
    out.append(f"  spurious (over-cut)  {d['spurious']:>7}   "
               "<- one document becomes two; shows up as missing fields")
    if d["same_type_boundaries"]:
        out.append("")
        out.append(f"  same-type joins      {d['same_type_boundaries']:>7}   "
                   f"recall {fmt(d['same_type_recall']).strip()}")
        out.append("     a page classifier cannot see these: the type never changes")
    return "\n".join(out)
