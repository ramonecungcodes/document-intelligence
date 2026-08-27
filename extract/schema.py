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
    # "including any prefix" used to end this line and meant the INV- in INV-4471.
    # A model reads it as "keep whatever precedes the number" and returns the
    # label word too, which is the opposite of what any identifier field wants.
    "identifier": "Identifier exactly as printed. Keep a prefix that is part of "
                  "the identifier itself, such as the INV- in INV-4471, but never "
                  "the label word printed in front of it.",
    "ssn": "Digits and separators exactly as printed.",
    "ein": "Digits and separators exactly as printed.",
    "phone": "Digits and separators exactly as printed.",
    "bool": "True if the box is checked or the answer is yes.",
}


def _property(spec: Field) -> dict:
    """A field's own description wins over the generic one for its kind.

    The generic text says how to format a value; only the field can say which value
    it wants. Where several fields share a kind, that distinction is the whole ball
    game -- three identifiers described identically get shuffled.
    """
    kind = JSON_TYPE.get(spec.kind, "string")
    prop = {"type": [kind, "null"]}
    description = getattr(spec, "help", "") or DESCRIPTIONS.get(spec.kind, "")
    if description and getattr(spec, "help", ""):
        generic = DESCRIPTIONS.get(spec.kind, "")
        description = f"{description} {generic}".strip()
    if description:
        prop["description"] = description
    return prop


ABSENCE_GUIDANCE = (
    "Decide first whether this document carries this field at all. Set status to "
    "'absent' and value to null when it does not -- that is a correct and expected "
    "answer, not a failure to find something. Set 'present' with the value only when "
    "the document actually shows it. Use 'unclear' when text might be it but you "
    "cannot tell. Never fill value by borrowing from another field or by inferring "
    "what a plausible answer would be."
)


def _optional_property(spec: Field, inner: dict) -> dict:
    """Ask whether the field exists before asking what it holds.

    A flat nullable field asks one question -- "what is the value?" -- and a required
    slot with no answer is pressure to invent one. Measured on the corpus: 46 of 46
    absent service locations were filled anyway, 37 of them copied verbatim from a
    neighbouring field, and a model four times larger produced exactly the same count.
    co_applicant_name did worse, inventing a person on all 25 loan applications that
    had none, without copying anything -- fabricating outright.

    Nullable was never the problem: the type already permitted null and the description
    already said null was usually right, three rewrites running, and nothing moved. So
    absence stops being a value the model may return and becomes a decision it has to
    make. The runner collapses the answer back to a plain value, leaving the record
    contract and every rule and scorer downstream untouched.
    """
    return {
        "type": "object",
        "description": (inner.get("description", "") + " " + ABSENCE_GUIDANCE).strip(),
        "properties": {
            "status": {
                "type": "string",
                "enum": ["present", "absent", "unclear"],
                "description": "Whether this document carries this field at all.",
            },
            "value": {k: v for k, v in inner.items() if k != "description"},
        },
        "required": ["status", "value"],
        "additionalProperties": False,
    }


def _object(fields, groups) -> dict:
    properties = {}
    for spec in fields:
        prop = _property(spec)
        properties[spec.name] = (_optional_property(spec, prop)
                                 if getattr(spec, "optional", False) else prop)
    for group in groups:
        properties[group.name] = _array(group)
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(group: Group) -> dict:
    """A repeating group.

    The group's own description matters most where rows are easy to confuse with
    something that merely looks like a row -- a totals line sits in the same table as
    the items it sums, and nothing but a description says which is which.
    """
    default = f"One entry per {group.name.replace('_', ' ').rstrip('s')} on the document."
    return {
        "type": "array",
        "description": getattr(group, "help", "") or default,
        # An unbounded array is an invitation to loop. Under constrained decoding the
        # model is never invalid while emitting one more element, so nothing forces it
        # to close the array -- it can only stop by exhausting max_tokens. One resume
        # did exactly that, repeating {"company": null, "title": "IT Manager",
        # "start_year": 2012} until it burned 49,853 characters, and no timeout could
        # catch it because a looping model streams steadily and never goes idle.
        #
        # The ceiling is deliberately far above anything real: the largest repeating
        # group anywhere in the corpus is six line items. This is not a modelling
        # decision about how many rows a document may have, it is a stop condition.
        "maxItems": getattr(group, "max_rows", 50),
        "items": _object(group.fields, group.groups),
    }


def json_schema(doctype: DocType, variant: str = "") -> dict:
    """The output schema for one document type, narrowed to a variant if it has them.

    The variant key is never asked for. It is what selected this schema in the first
    place, so requesting it back is circular -- and the model cannot answer it anyway:
    "onboarding" is a name in our taxonomy, not a string printed on the page. Asked for
    a form_type, a model reasonably returns what the form actually calls itself,
    "HR-ONB-1002" or "New Hire Onboarding Form", and is marked wrong for being right.
    Measured on the first baseline: 160 of 160 forms wrong on that one field.

    The runner supplies it from the corpus instead. Deciding it from the document is
    classification, which is the classifier slot's job, not something to smuggle into
    the extractor's schema.
    """
    return _object(doctype.graded_fields(variant), doctype.groups)


def instructions(doctype: DocType, variant: str = "") -> str:
    """Field-level guidance appended to the system prompt.

    Kept short on purpose: the schema already carries the structure, and a long
    restatement of it competes with the document for the model's attention.
    """
    label = doctype.name.replace("_", " ")
    if variant:
        label = f"{variant.replace('_', ' ')} {label}"
    lines = [
        f"You are extracting the fields of a {label}.",
        "",
        "Rules:",
        "- Copy values exactly as they appear on the document. Do not reformat dates,",
        "  identifiers, or phone numbers.",
        "- Amounts are numbers: strip currency symbols and thousands separators.",
        "- If a field is genuinely not on the document, return null. Never guess, and",
        "  never carry a value over from a different field because it looks plausible.",
        "- Return the value only, not the label printed before it. A document showing",
        "  `METER M3947745` has a meter number of `M3947745`; `SITE 12 Oak St` is an",
        "  address of `12 Oak St`.",
        "- Transcribe what is printed even when it looks wrong. If the stated total",
        "  disagrees with the line items, return the stated total; detecting that",
        "  disagreement is somebody else's job.",
    ]
    for group in doctype.groups:
        detail = getattr(group, "help", "")
        lines.append(f"- `{group.name}`: one entry per row, in the order they appear. "
                     f"Include every row." + (f" {detail}" if detail else ""))
        for nested in group.groups:
            nested_detail = getattr(nested, "help", "")
            lines.append(f"- `{group.name}[].{nested.name}`: likewise."
                         + (f" {nested_detail}" if nested_detail else ""))
    if any(g.name == "sections" for g in doctype.groups):
        lines += [
            "",
            "This document bills several services that are paid separately. Each one has",
            "its own account number, service period and total, and each must appear as its",
            "own entry in `sections`. Missing one means an invoice that cannot be paid.",
        ]
    return "\n".join(lines)
