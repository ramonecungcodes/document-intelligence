"""Turn a document type declaration into a JSON Schema for structured output.

This is where the type registry starts paying for itself. `core/doctypes.py` already
says what fields an invoice has and how each is compared; the same declaration now
generates the schema the model is constrained to, so the extractor and the scorer can
never disagree about what an invoice is.

Every field is nullable and every field is required. Structured outputs demands the
full `required` list plus `additionalProperties: false`, and the corpus contains
fields that are legitimately absent -- a W-9 has an SSN or an EIN, never both, and the
defective set empties fields on purpose. Nullable-and-required lets the model say
"not present" without inventing a value or dropping the key.
"""
from __future__ import annotations

from core.doctypes import DocType, Field, Group

# How a declared field kind is represented on the wire. Everything the scorer
# normalises from text (dates, identifiers, phone numbers) stays a string: asking the
# model to reformat a date is asking it to make a second mistake.
JSON_TYPE = {
    "money": "number",
    "number": "number",
    "bool": "boolean",
}

DESCRIPTIONS = {
    "date": "Date exactly as printed on the document.",
    "money": "Numeric amount only, no currency symbol or thousands separators.",
    "identifier": "Identifier exactly as printed, including any prefix.",
    "ssn": "Digits and separators exactly as printed.",
    "ein": "Digits and separators exactly as printed.",
    "phone": "Digits and separators exactly as printed.",
    "bool": "True if the box is checked or the answer is yes.",
}


def _property(spec: Field) -> dict:
    kind = JSON_TYPE.get(spec.kind, "string")
    prop = {"type": [kind, "null"]}
    if spec.kind in DESCRIPTIONS:
        prop["description"] = DESCRIPTIONS[spec.kind]
    return prop


def _object(fields, groups) -> dict:
    properties = {spec.name: _property(spec) for spec in fields}
    for group in groups:
        properties[group.name] = _array(group)
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(group: Group) -> dict:
    return {
        "type": "array",
        "description": f"One entry per {group.name.replace('_', ' ').rstrip('s')} on the document.",
        "items": _object(group.fields, group.groups),
    }


def json_schema(doctype: DocType) -> dict:
    """The output schema for one document type."""
    return _object(doctype.fields, doctype.groups)


def instructions(doctype: DocType) -> str:
    """Field-level guidance appended to the system prompt.

    Kept short on purpose: the schema already carries the structure, and a long
    restatement of it competes with the document for the model's attention.
    """
    lines = [
        f"You are extracting the fields of a {doctype.name.replace('_', ' ')}.",
        "",
        "Rules:",
        "- Copy values exactly as they appear on the document. Do not reformat dates,",
        "  identifiers, or phone numbers.",
        "- Amounts are numbers: strip currency symbols and thousands separators.",
        "- If a field is genuinely not on the document, return null. Never guess, and",
        "  never carry a value over from a different field because it looks plausible.",
        "- Transcribe what is printed even when it looks wrong. If the stated total",
        "  disagrees with the line items, return the stated total; detecting that",
        "  disagreement is somebody else's job.",
    ]
    if doctype.groups:
        for group in doctype.groups:
            lines.append(
                f"- `{group.name}`: one entry per row, in the order they appear. "
                f"Include every row."
            )
    if any(g.name == "sections" for g in doctype.groups):
        lines += [
            "",
            "This document bills several services that are paid separately. Each one has",
            "its own account number, service period and total, and each must appear as its",
            "own entry in `sections`. Missing one means an invoice that cannot be paid.",
        ]
    return "\n".join(lines)
