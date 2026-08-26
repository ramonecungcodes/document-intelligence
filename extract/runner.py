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
from extract.text import read_pdf


@dataclass
class Result:
    record: dict
    usage: Usage = field(default_factory=Usage)
    error: str = ""
    skipped: str = ""
    rules: object = None        # RuleReport: what deterministic cleanup changed


def extract_document(backend, doctype: DocType, pdf_path: str, relative_path: str,
                     variant: str = "", rule_settings=None) -> Result:
    """Read one PDF and return a prediction record shaped like a corpus label.

    `variant` narrows the schema for types that have them -- a W-9 is asked for the ten
    fields a W-9 has, not the sixty-four spanning every kind of form. Like the document
    type itself it is given rather than predicted; classification is a later phase.
    """
    page_text = read_pdf(pdf_path)
    base = {"file": relative_path, "doc_type": doctype.name}
    if variant and doctype.variant_key:
        base[doctype.variant_key] = variant

    if page_text.empty:
        # No text layer: the honest answer is that this extractor cannot read it.
        base["_note"] = "no text layer"
        return Result(record=base, skipped="no text layer")

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
        # Structured output should make this impossible; when a backend's schema
        # support is partial it is the first thing to break, so say so plainly.
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
