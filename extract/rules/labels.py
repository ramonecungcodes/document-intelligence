"""Remove a printed label the model transcribed along with the value beside it.

Documents print `METER M3947745`, `Bill of lading C-59602`, `CIRCUIT EQP/24046/DS1`.
The label names the value; it is not part of it. Asked for the meter number, a model
that has correctly located the right region on the page frequently returns the whole
phrase -- the value is right, the span is one word too wide.

That is wrong by definition rather than by judgement, which is what makes it a rule:
an identifier is a single token. It never contains a space. Measured across the whole
corpus, 701 identifier values, none carried one. So a multi-token identifier is not an
identifier that happens to look unusual; it is an identifier with something else stuck
to the front, and the only question is where to cut.

The cut is deliberately conservative. Every token before the last must be purely
alphabetic and the last must contain a digit. `Bill of lading C-59602` splits; a value
like `C-59602 A1` does not, because the leading token is not a word. If the shape is
at all ambiguous the value is left alone and scored as the model returned it.

Scope comes from the type registry rather than a list of field names kept here: the
rule asks the record which type it is, asks the registry which of that type's fields
are declared `identifier`, and touches only those. A new identifier field anywhere in
any type is covered the day it is declared, and a text field never is.
"""
from __future__ import annotations

from core.doctypes import REGISTRY, DocType
from core.rules import RULES


def _identifier_fields(doctype: DocType, variant: str = "") -> set:
    """Every field declared `identifier`, including nested groups and variants."""
    names = set()

    def walk(fields, groups):
        names.update(spec.name for spec in fields if spec.kind == "identifier")
        for group in groups:
            walk(group.fields, group.groups)

    walk(doctype.fields_for(variant) if variant else doctype.fields, doctype.groups)
    # A rule may run before the variant is known; cover every variant's fields so a
    # narrowed schema is never the reason a label survives.
    for fields in (doctype.variants or {}).values():
        walk(fields, [])
    return names


def _strip(value: str) -> str:
    """`Bill of lading C-59602` -> `C-59602`. Anything ambiguous is returned as-is."""
    tokens = value.split()
    if len(tokens) < 2:
        return value
    if not all(token.isalpha() for token in tokens[:-1]):
        return value
    if not any(character.isdigit() for character in tokens[-1]):
        return value
    return tokens[-1]


@RULES.register(
    "strip_identifier_labels",
    help="Drop a label word transcribed in front of an identifier (`METER M394` -> `M394`).",
)
def strip_identifier_labels(record: dict) -> int:
    doctype = REGISTRY.get(record.get("doc_type"))
    if doctype is None:
        return 0
    variant = record.get(doctype.variant_key) if doctype.variant_key else ""
    names = _identifier_fields(doctype, variant if isinstance(variant, str) else "")
    if not names:
        return 0

    changed = 0

    def clean(container: dict) -> None:
        nonlocal changed
        for name in names:
            value = container.get(name)
            if not isinstance(value, str):
                continue
            stripped = _strip(value.strip())
            if stripped != value:
                container[name] = stripped
                changed += 1
        for group in ("sections", "line_items", "work_history"):
            for row in container.get(group) or []:
                if isinstance(row, dict):
                    clean(row)

    clean(record)
    return changed
