"""One document in, one structured extraction out.

Deliberately the crudest thing that can produce a number: read the text layer, send it
once with a schema, keep what comes back. No tools, no retries on content, no repair
loop, no confidence. Those are later phases, and each has to justify itself against
whatever this scores.

The document type is taken from the corpus rather than predicted. Phase 1 is measuring
whether a model can read fields off a document it has never seen; classification is a
separate risk with its own phase, and mixing them would make a bad number impossible
to attribute.

Which model answers is a backend concern (see `backends.py`) -- this file does not
know or care whether it is talking to a frontier API or a laptop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from core.doctypes import DocType
from extract import schema as schema_mod
from extract.rules import RULES
from extract.backends import Usage
from extract.text import read_pdf   # the default normalizer, for callers that pass none


@dataclass
class Result:
    record: dict
    usage: Usage = field(default_factory=Usage)
    error: str = ""
    skipped: str = ""
    rules: object = None        # RuleReport: what deterministic cleanup changed


def collapse_optional(record: dict, doctype, variant: str = "") -> None:
    """Flatten {"status": ..., "value": ...} back to a plain value, in place.

    Fields declared `optional` are asked as a decision rather than a slot, so the model
    answers with a shape. Nothing downstream should have to know that: the rules, the
    scorer and the stored record all speak plain values, and making them handle two
    shapes would spread one extraction detail across the whole system.

    'present' keeps the value, 'absent' and 'unclear' become None. Treating 'unclear'
    as absent is deliberate for now -- this field is worth optimising for precision,
    since an invented address flows downstream unchallenged while a blank one gets
    looked at. When confidence routing exists, 'unclear' is what feeds it, and that is
    the reason it is a distinct answer rather than folded into 'absent' at the model.
    """
    optional = {spec.name for spec in doctype.graded_fields(variant)
                if getattr(spec, "optional", False)}
    for group in doctype.groups:
        optional |= {spec.name for spec in group.fields
                     if getattr(spec, "optional", False)}

    def walk(container):
        if not isinstance(container, dict):
            return
        for name in list(container):
            answer = container.get(name)
            if name in optional and isinstance(answer, dict) and "status" in answer:
                container[name] = (answer.get("value")
                                   if answer.get("status") == "present" else None)
            elif isinstance(answer, list):
                for row in answer:
                    walk(row)

    walk(record)


def extract_document(backend, doctype: DocType, pdf_path: str, relative_path: str,
                     variant: str = "", rule_settings=None, normalizer=None) -> Result:
    """Read one PDF and return a prediction record shaped like a corpus label.

    `variant` narrows the schema for types that have them -- a W-9 is asked for the ten
    fields a W-9 has, not the sixty-four spanning every kind of form. Like the document
    type itself it is given rather than predicted; classification is a later phase.
    """
    # Whoever asked for this extraction decides how the text is obtained. Defaulting to
    # the embedded text layer keeps Phase 1 behaviour for every caller that does not
    # care, while a degraded run supplies the `cached` normalizer and reads OCR output
    # this file never has to know about.
    page_text = normalizer.read(pdf_path) if normalizer is not None else read_pdf(pdf_path)
    base = {"file": relative_path, "doc_type": doctype.name}
    if variant and doctype.variant_key:
        base[doctype.variant_key] = variant

    if page_text.empty:
        # Nothing to read. Under the native reader that means the PDF has no text
        # layer; under an OCR normalizer it means the engine recovered nothing from the
        # page. Recording which is which matters -- they are different findings, and a
        # run that conflates them cannot say whether OCR helped.
        base["_note"] = f"no text ({page_text.engine or 'native'})"
        base["_normalizer"] = page_text.provenance()
        return Result(record=base, skipped="no text")

    # How the text was obtained travels with the prediction. An accuracy figure on
    # degraded documents means something very different depending on which engine read
    # them, and a number whose provenance is unknown is not a measurement.
    base["_normalizer"] = page_text.provenance()

    completion = backend.complete(
        system=schema_mod.instructions(doctype, variant),
        user=f"Document text:\n\n{page_text.text}",
        schema=schema_mod.json_schema(doctype, variant),
    )
    if completion.error:
        base["_error"] = completion.error
        return Result(record=base, usage=completion.usage, error=completion.error)

    try:
        parsed = json.loads(completion.text)
    except json.JSONDecodeError as error:
        # Two very different faults land here and must not be reported alike.
        #
        # A truncated answer is a budget problem: the model was still writing when it
        # hit max_tokens, so the JSON is valid right up to the cut. Every one of the
        # twelve resumes that failed the first full corpus run was this, and each was
        # reported as "unparseable JSON" -- which reads like a broken model or a broken
        # schema and sends you looking in entirely the wrong place. The fix is a bigger
        # budget, so the message has to say so.
        if completion.truncated:
            detail = (f"ran out of tokens mid-answer after {len(completion.text)} "
                      f"characters; raise max_tokens")
            base["_error"] = f"truncated: {detail}"
            return Result(record=base, usage=completion.usage, error=f"truncated: {detail}")

        # Anything still here is genuinely malformed rather than merely cut off, which
        # is what a backend with only partial schema support looks like.
        preview = completion.text[:120].replace("\n", " ")
        base["_error"] = f"unparseable JSON: {error}"
        return Result(record=base, usage=completion.usage,
                      error=f"unparseable JSON ({error}): {preview!r}")

    if not isinstance(parsed, dict):
        base["_error"] = f"expected an object, got {type(parsed).__name__}"
        return Result(record=base, usage=completion.usage,
                      error=f"expected an object, got {type(parsed).__name__}")

    if completion.truncated:
        base["_note"] = "truncated at max tokens"
    collapse_optional(parsed, doctype, variant)
    base.update(parsed)
    # The variant key is ours, not the model's. It is excluded from the schema, but a
    # backend in prompt mode can return whatever it likes, and update() would let that
    # overwrite the value the corpus already told us.
    if variant and doctype.variant_key:
        base[doctype.variant_key] = variant

    # Deterministic cleanup, declared per document type. What each rule changed is
    # recorded on the prediction rather than absorbed, so a run can never quietly
    # flatter its own numbers.
    report = RULES.apply(base, doctype.name, rule_settings)
    if report.total:
        base["_rules_applied"] = report.to_dict()
    return Result(record=base, usage=completion.usage, rules=report)
