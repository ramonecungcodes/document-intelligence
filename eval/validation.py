"""Scoring for the validator stage, and the self-test that has to pass first.

Two numbers, and running them in the wrong order makes both meaningless.

**Rule correctness**, measured against the corpus labels. The labels are what the
document actually says, so a rule that fires on a clean document's ground truth is
simply wrong -- there is no extraction to blame. This has to come out with zero false
positives before a rule is allowed near a prediction, which is the same bar Phase 0 set
when it demanded `score --predictions self` return exactly 1.000.

**Defect detection**, measured against extracted output. This is the product of the
rule and the extractor, and it is the number a person actually experiences. It is only
interpretable once rule correctness is clean, because otherwise a false positive has
two possible parents and no way to tell them apart.

Per-code precision and recall, not just totals. The families fail differently: an
arithmetic rule that misses nothing and a required-field rule that fires constantly
average out to something respectable and describe neither. And a defect class the
corpus carries but no rule detects has to be visible as a zero rather than absent from
the table -- a suite that silently declines to check something reads exactly like a
suite that checks it and finds nothing.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


def _rate(numerator: int, denominator: int):
    return round(numerator / denominator, 4) if denominator else None


@dataclass
class ValidationScore:
    """Agreement between the defects found and the defects injected."""

    hit: dict = field(default_factory=lambda: defaultdict(int))
    missed: dict = field(default_factory=lambda: defaultdict(int))
    spurious: dict = field(default_factory=lambda: defaultdict(int))
    documents: int = 0
    clean_documents: int = 0
    clean_documents_flagged: int = 0
    defective_documents: int = 0
    defective_documents_flagged: int = 0
    # Codes the corpus injects, so a class nobody checks shows as a zero row rather
    # than vanishing from the report.
    known: set = field(default_factory=set)

    def add(self, found: set, injected: set, is_clean: bool = False) -> None:
        self.documents += 1
        self.known |= injected
        for code in found & injected:
            self.hit[code] += 1
        for code in injected - found:
            self.missed[code] += 1
        for code in found - injected:
            self.spurious[code] += 1
        if is_clean:
            self.clean_documents += 1
            self.clean_documents_flagged += bool(found)
        else:
            self.defective_documents += 1
            self.defective_documents_flagged += bool(found)

    def per_code(self) -> list:
        rows = []
        for code in sorted(self.known | set(self.spurious)):
            hit, missed, spurious = (self.hit[code], self.missed[code],
                                     self.spurious[code])
            rows.append({
                "code": code,
                "injected": hit + missed,
                "found": hit + spurious,
                "recall": _rate(hit, hit + missed),
                "precision": _rate(hit, hit + spurious),
            })
        return rows

    def to_dict(self) -> dict:
        hit = sum(self.hit.values())
        missed = sum(self.missed.values())
        spurious = sum(self.spurious.values())
        precision = _rate(hit, hit + spurious)
        recall = _rate(hit, hit + missed)
        return {
            "documents": self.documents,
            "defects_injected": hit + missed,
            "defects_found": hit + spurious,
            "precision": precision,
            "recall": recall,
            "f1": (round(2 * precision * recall / (precision + recall), 4)
                   if precision and recall else None),
            # What routing actually needs: was this document flagged at all. Several
            # of the corpus's codes are different *causes* of one observable -- a
            # tampered tax and a tampered total both present as subtotal + tax != total
            # and nothing on the page says which -- so per-code recall understates a
            # validator that correctly caught the document and named a sibling code.
            "defective_documents": self.defective_documents,
            "defective_documents_caught": self.defective_documents_flagged,
            "document_recall": _rate(self.defective_documents_flagged,
                                     self.defective_documents),
            # The number that decides whether a rule may ship. A clean document that
            # trips a rule on its own ground truth is the rule being wrong.
            "clean_documents": self.clean_documents,
            "clean_documents_flagged": self.clean_documents_flagged,
            "false_alarm_rate": _rate(self.clean_documents_flagged,
                                      self.clean_documents),
            "per_code": self.per_code(),
        }


def render(score: ValidationScore, against: str = "labels") -> str:
    d = score.to_dict()
    truth = against == "labels"

    def fmt(v, w=8):
        return f"{'--':>{w}}" if v is None else f"{v:>{w}.3f}"

    out = ["", f"VALIDATION - against {against}", ""]
    out.append(f"  documents            {d['documents']:>8}")
    out.append(f"  defects injected     {d['defects_injected']:>8}")
    out.append(f"  defects found        {d['defects_found']:>8}")
    out.append(f"  precision           {fmt(d['precision'])}")
    out.append(f"  recall              {fmt(d['recall'])}")
    out.append(f"  F1                  {fmt(d['f1'])}")
    if d["defective_documents"]:
        out.append("")
        out.append(f"  defective documents  {d['defective_documents']:>8}")
        out.append(f"  of those, caught     {d['defective_documents_caught']:>8}   "
                   f"document recall {fmt(d['document_recall']).strip()}")
        out.append("     several codes are causes of one observable; this is what "
                   "routing needs")
    if d["clean_documents"]:
        out.append("")
        out.append(f"  clean documents      {d['clean_documents']:>8}")
        out.append(f"  of those, flagged    {d['clean_documents_flagged']:>8}   "
                   + ("<- MUST be 0: on ground truth there is no extractor to blame"
                      if truth else "<- rule or extractor; only readable once the "
                                    "self-test is clean"))
    out.append("")
    out.append(f"  {'defect':<38}{'n':>5}{'found':>7}{'recall':>9}{'precision':>11}")
    for row in d["per_code"]:
        out.append(f"  {row['code']:<38}{row['injected']:>5}{row['found']:>7}"
                   f"{fmt(row['recall'], 9)}{fmt(row['precision'], 11)}")
    return "\n".join(out)
