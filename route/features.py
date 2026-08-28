"""Signals about a document that do not come from asking the model how it did.

Phase 5's wording is "calibration from independent signals rather than model
self-report", and the measurement that made the distinction concrete is in
`reports/calibration-extraction.json`: the classifier's own confidence, which is well
calibrated for classification at ECE 0.063, carries no usable information about whether
the extraction was right, and pooled across types it points the wrong way. It is a
statement about how visually distinctive the page is. Nothing about that predicts
whether 24 fields came back correct.

So the signals here are all of a different kind. Each is something observable about the
document or about the *shape* of the answer, never the model's opinion of its own work:

    how well the page was read     mean OCR word confidence, word count, which layer
    whether the readers agreed     the two OCR engines' character counts, from `tried`
    how much came back empty       the share of expected fields the extractor left blank
    what the rules found           validator errors and warnings, re-run deterministically
    how ragged the answer was      how much deterministic cleanup it needed

Two of these deserve their reasoning stated, because both look like the model's opinion
and are not.

Blank share is the extractor declining, not the extractor scoring itself. A model that
returns nothing for eleven of twenty fields has told you something checkable about the
page. It has not told you it is unsure.

Validator findings are re-run here rather than read from a stored file, because they are
deterministic given the record and the rules -- the same reasoning that kept them out of
the signals sidecar. Storing them would have made a second copy of a fact that the rules
can always reproduce, and copies drift.

Everything returned is either a float or None. None means the signal was not available
for that document -- a native-read page has no OCR confidence, and no page has a
confidence of zero. Substituting zero would tell a calibrator that the cleanest
documents in the corpus were the least legible.
"""
from __future__ import annotations

import os
import re

# `tesseract=33ch/conf0.78/1.1s` -- what the cascade normalizer records about each
# engine it tried, which is the only place the losing engine's opinion survives.
TRIED = re.compile(r"(?P<engine>\w+)=(?P<chars>\d+)ch/conf(?P<conf>[\d.]+)")


def profile_of(relative_path: str) -> str:
    """Which degradation this document is, read off its name."""
    stem = os.path.basename(relative_path)
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    _, sep, profile = stem.partition("__")
    return profile if sep else "clean"


def _engines(tried) -> dict:
    """`['tesseract=33ch/conf0.78/1.1s', ...]` -> {engine: (chars, confidence)}."""
    out = {}
    for entry in tried or []:
        found = TRIED.search(str(entry))
        if found:
            out[found.group("engine")] = (int(found.group("chars")),
                                          float(found.group("conf")))
    return out


def reading(record: dict) -> dict:
    """How well the page was read, from the normalizer's own provenance.

    `agreement` here is between two OCR engines, not between a model and itself. When
    two independent readers disagree about how much text is on a page, one of them is
    wrong about the page, and which one is not knowable from here -- but that the
    disagreement exists is knowable, and it is exactly the kind of signal the phase asks
    for. It is expressed as the smaller character count over the larger, so 1.0 is
    perfect agreement and 0.0 is one engine finding nothing.
    """
    block = record.get("_normalizer") or {}
    out = {
        "ocr_confidence": block.get("confidence"),
        "words": (float(block["words"]) if block.get("words") is not None else None),
        "pages": (float(block["pages"]) if block.get("pages") is not None else None),
        "is_ocr": 1.0 if block.get("layer") == "ocr" else 0.0,
    }
    engines = _engines(block.get("tried"))
    counts = sorted(chars for chars, _ in engines.values())
    if len(counts) > 1:
        out["engine_agreement"] = (round(counts[0] / counts[-1], 4)
                                   if counts[-1] else 0.0)
        confidences = [conf for _, conf in engines.values()]
        out["engine_confidence_spread"] = round(max(confidences) - min(confidences), 4)
    else:
        out["engine_agreement"] = None
        out["engine_confidence_spread"] = None
    # Words per page rather than words, because a two-page document legitimately holds
    # twice the text and a raw count would rank it as twice as legible.
    if out["words"] is not None and out["pages"]:
        out["words_per_page"] = round(out["words"] / out["pages"], 2)
    else:
        out["words_per_page"] = None
    return out


def answer_shape(record: dict, doctype, variant: str = "") -> dict:
    """What the extractor returned, measured without asking it anything.

    The share of expected fields left blank is the signal that matters here, and it is
    counted against the schema rather than against the keys the model happened to emit.
    Counting emitted keys would score a model that returned three fields out of twenty
    as 100% complete, which is precisely backwards -- an answer that omits most of the
    document is the case this is meant to catch.
    """
    expected = list(doctype.graded_fields(variant)) if doctype else []
    blank = 0
    for spec in expected:
        value = record.get(spec.name)
        if value is None or (isinstance(value, str) and not value.strip()):
            blank += 1
    out = {
        "expected_fields": float(len(expected)) or None,
        "blank_share": (round(blank / len(expected), 4) if expected else None),
    }
    rows = 0
    for group in (doctype.groups if doctype else []):
        value = record.get(group.name)
        if isinstance(value, list):
            rows += len(value)
    out["rows"] = float(rows) if any(
        isinstance(record.get(g.name), list) for g in (doctype.groups if doctype else [])
    ) else None
    return out


def rule_activity(record: dict) -> dict:
    """How much deterministic cleanup the answer needed.

    A proxy for raggedness, and one that is invisible in the cleaned record -- which is
    the point of cleaning it. A model that emitted rollup rows and empty rows was
    producing something structurally further from the schema than one that did not, and
    that is a fact about the answer rather than a claim by its author.
    """
    block = record.get("_rules") or {}
    applied = block.get("applied") if isinstance(block, dict) else None
    if not isinstance(applied, dict):
        return {"rules_applied": None}
    return {"rules_applied": float(sum(applied.values()))}


def findings(record: dict, validators, doctype, variant: str = "") -> dict:
    """What the rules say about this record, re-run rather than read from a file.

    Deterministic given the record and the rule set, which is why it was deliberately
    kept out of the signals sidecar: a stored copy of a reproducible fact is a copy that
    can go stale while looking current.

    Errors and warnings are counted apart. A warning is a rule saying a person should
    look; an error is a rule saying the document does not add up. Summing them would
    let five soft observations outrank one arithmetic failure.
    """
    from validate.base import run as run_validators

    if not validators or doctype is None:
        return {"validator_errors": None, "validator_warnings": None}
    report = run_validators(validators, record, doctype, variant)
    errors = sum(1 for f in report.findings if f.severity == "error")
    return {
        "validator_errors": float(errors),
        "validator_warnings": float(len(report.findings) - errors),
    }


def self_report(signals_row: dict) -> dict:
    """What the model said about itself, carried alongside the rest as the control.

    It is here to be beaten, not to be used. Every other signal in this module is
    something observable about the document; this one is the classifier's own opinion,
    and the whole phase turns on whether the observable ones do better. Scoring it in
    the same table on the same documents is the only way that comparison means
    anything -- measured on two different corpora it would be an anecdote.

    `margin` is kept separate from `confidence` because they are different claims. A
    model at 0.95 with nothing near it and one at 0.95 with 0.94 behind it are in
    different states, and only the second is a near-tie.
    """
    block = (signals_row or {}).get("classifier") or {}
    return {
        "classifier_confidence": block.get("confidence"),
        "classifier_margin": block.get("margin"),
    }


# Ordered so a report reads from "could the page be read" through "what came back" to
# "does it hold together", and ends with the control -- the order a person would ask
# them in, with the thing being argued against last.
NAMES = (
    "ocr_confidence",
    "words_per_page",
    "engine_agreement",
    "engine_confidence_spread",
    "is_ocr",
    "blank_share",
    "rows",
    "rules_applied",
    "validator_errors",
    "validator_warnings",
    "classifier_confidence",
    "classifier_margin",
)

# Which direction a *higher* value is expected to mean a better extraction. Recorded
# rather than inferred so a signal that comes out backwards is visible as a surprise
# instead of being quietly re-read as confirmation.
EXPECTED = {
    "ocr_confidence": +1,
    "words_per_page": +1,
    "engine_agreement": +1,
    "engine_confidence_spread": -1,
    "is_ocr": -1,
    "blank_share": -1,
    "rows": 0,
    "rules_applied": -1,
    "validator_errors": -1,
    "validator_warnings": -1,
    # Expected positive, which is exactly the assumption the phase set out to test.
    # Declaring it here means a negative result is reported as a contradicted
    # expectation rather than silently re-oriented into a success.
    "classifier_confidence": +1,
    "classifier_margin": +1,
}


def extract(record: dict, doctype, variant: str = "", validators=None,
            signals_row: dict = None) -> dict:
    """Every signal this module can produce for one prediction record.

    `signals_row` is the sidecar entry when one exists. Runs made before it did have
    none, and those documents get None for the classifier signals rather than being
    dropped -- a signal missing for a document is a different fact from a document
    missing, and only the first should shrink one column of the table.
    """
    out = {}
    out.update(reading(record))
    out.update(answer_shape(record, doctype, variant))
    out.update(rule_activity(record))
    out.update(findings(record, validators, doctype, variant))
    out.update(self_report(signals_row))
    return {name: out.get(name) for name in NAMES}
