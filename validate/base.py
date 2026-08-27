"""What a validator returns, and the trap this whole stage has to be built around.

A validator reads an extracted document and says what is wrong with it: the line items
do not sum to the subtotal, the SSN is not nine digits, the loss was reported before it
happened. The corpus carries 352 documents with 527 such defects deliberately injected
and tagged, so the stage can be scored rather than admired.

The trap is that a validator runs on *extracted* output, and extraction is imperfect.
When a rule fires there are two explanations and they demand opposite responses:

    the document really is defective          -> flag it, route it, this is the job
    the extractor misread a good document     -> fix extraction; the rule is innocent

Nothing in the firing distinguishes them. A stage that cannot tell those apart reports
a defect-detection rate that is partly its own extraction error, and the number moves
when the model changes for reasons that have nothing to do with the rules.

So every validator here is scored twice, and the first run is the one that matters.

    against the ground truth labels   the rule's own correctness. A rule that fires on
                                      a clean document's *truth* is simply wrong, and
                                      no amount of extraction quality will save it.
                                      This must come out perfect before the rule ships.

    against extracted output          the pipeline's defect detection, which is the
                                      product of the rule and the extractor, and is
                                      only interpretable once the first number is 1.000.

That is the same discipline Phase 0 applied to scoring, where `score --predictions
self` has to return exactly 1.000 before any model number means anything. A validator
suite with no self-test is a suite that grades the extractor and calls it a defect rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Finding:
    """One thing wrong with one document."""

    code: str                            # matches a corpus defect tag where one exists
    field: str = ""                      # which field, when it is about one
    message: str = ""                    # what a person needs to read
    expected: Optional[str] = None       # what the rule computed
    actual: Optional[str] = None         # what the document said
    severity: str = "error"              # error | warning
    validator: str = ""

    def to_dict(self) -> dict:
        return {"code": self.code, "field": self.field or None,
                "message": self.message, "expected": self.expected,
                "actual": self.actual, "severity": self.severity,
                "validator": self.validator}


@dataclass
class Report:
    """Everything wrong with one document, and what looked at it."""

    findings: list = field(default_factory=list)
    checked: list = field(default_factory=list)     # validators that ran
    skipped: list = field(default_factory=list)     # and why they could not
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    @property
    def codes(self) -> set:
        return {f.code for f in self.findings}

    def to_dict(self) -> dict:
        return {"ok": self.ok,
                "findings": [f.to_dict() for f in self.findings],
                "checked": self.checked,
                "skipped": self.skipped or None,
                "seconds": round(self.seconds, 3) or None}


class Validator:
    """A rule, or a family of them, over one extracted document.

    `applies_to` keeps a rule from firing on a type it was never written for. A
    validator that runs everywhere and returns nothing for most types looks the same as
    one that is broken, and neither is visible until something is silently not checked.
    """

    name = ""
    SETTINGS: tuple = ()
    applies_to: tuple = ()          # doc type names; empty means every type

    def handles(self, doc_type: str) -> bool:
        return not self.applies_to or doc_type in self.applies_to

    def check(self, record: dict, doctype, variant: str = "") -> list:
        raise NotImplementedError


VALIDATORS: dict = {}


def register(name: str):
    def wrap(cls):
        if name in VALIDATORS:
            raise ValueError(f"duplicate validator {name!r}")
        cls.name = name
        VALIDATORS[name] = cls
        return cls
    return wrap


def build_all(config=None, enabled=None):
    """Every validator that should run, in a stable order.

    Ordered by name rather than registration, so a report reads the same way twice and
    a diff between two runs is about the documents rather than about import order.
    """
    from core import config as config_mod

    config = config or config_mod.load()
    chosen = enabled if enabled is not None else sorted(VALIDATORS)
    out = []
    for name in sorted(chosen):
        if name not in VALIDATORS:
            raise SystemExit(
                f"unknown validator {name!r}; available: {', '.join(sorted(VALIDATORS))}")
        cls = VALIDATORS[name]
        settings = config.settings("validator", name, cls.SETTINGS) if cls.SETTINGS else {}
        out.append(cls(**settings))
    return out


def run(validators, record: dict, doctype, variant: str = "") -> Report:
    """Every applicable rule over one document."""
    import time

    started = time.time()
    report = Report()
    for validator in validators:
        if not validator.handles(doctype.name):
            report.skipped.append(f"{validator.name}: not for {doctype.name}")
            continue
        found = validator.check(record, doctype, variant) or []
        for finding in found:
            finding.validator = validator.name
        report.findings.extend(found)
        report.checked.append(validator.name)
    report.seconds = time.time() - started
    return report
