"""Template for a post-extraction rule, and the contract for the record it receives.

Copy this file, rename it, write your function, and import it in `__init__.py`.
Nothing here is registered, so this module has no effect on a run -- it exists to be
read and copied.


WHAT A RULE IS
--------------
A deterministic transformation applied to one prediction after the model returns and
before it is scored or stored. A rule fixes something wrong *by definition* -- the kind
of thing a validator would reject as structurally impossible.

    a totals row is not a line item
    a row with no description, amount or quantity is not a row

A rule must never correct the model's judgement. If a stated total disagrees with its
line items, that is a finding for the validators and precisely what the defect corpus
exists to measure; repairing it here would make the extractor look better, the system
worse, and the evaluation meaningless.

The test: could you state the fix as a fact about documents, without referring to what
the model probably meant? If not, it is not a rule.


THE RECORD
----------
A plain dict, the model's answer for one document, shaped like a corpus label so it can
be scored by joining on `file`. Mutate it in place and return how many things you
changed.

ALWAYS PRESENT
    file        str   corpus-relative path, e.g. "invoices/northwind_INV-20261000.pdf".
                      The join key. Never rename or remove it.
    doc_type    str   "invoice", "purchase_order", "multi_bill_invoice", "resume",
                      "form". Same value the registry dispatched on, so a scoped rule
                      can trust it.

PRESENT FOR TYPES THAT HAVE VARIANTS
    form_type   str   "onboarding" | "claim" | "w9" | "w4" | "loan", on forms only.
                      Set from the corpus, not predicted -- see DocType.variant_key.

EVERYTHING ELSE IS OPTIONAL. Assume nothing.
    Extracted fields are whatever the model returned. Two reasons a field you expect
    may be absent:

      the schema is narrowed by variant -- a W-9 record has `ssn`, `ein` and
      `tax_classification`; a loan record has none of them and carries `loan_amount`
      instead. Measured on a real run: of 160 form records, `ssn` appeared in 120 and
      `tax_classification` in 20.

      the model omitted it. Every field in the schema is nullable, and where an
      endpoint cannot do constrained decoding the schema is carried in the prompt and
      compliance is not guaranteed.

    So a present key may still hold None, and an absent key is normal. Use
    `record.get(name)` and check for None. Never `record[name]`.

NESTED GROUPS -- also optional, and may be an empty list
    line_items  list[dict]  on invoices and purchase orders, and inside each section
                            of a multi-bill invoice
    sections    list[dict]  multi-bill invoices only; each section may itself carry
                            line_items
    work_history list[dict] resumes only

    Walk them defensively:  for row in record.get("line_items") or []:

NEVER PRESENT
    _error      A rule never sees a failed extraction. Parse failures, refusals and
                transport errors return before the rules run, so if you are holding a
                record, the model answered and the answer parsed.

    irregularities, layout, degradation, source_file
                Ground-truth and corpus metadata. This is a prediction; it has never
                been near a label. A rule cannot see the right answer, which is
                deliberate -- a rule that could would be cheating.

ADDED AFTER YOU RUN
    _rules_applied  dict  {rule_name: count}, written by the registry from what each
                          rule returned. Do not set it yourself.


TYPES ARE NOT GUARANTEED
------------------------
The schema asks for numbers on money and quantity fields, but a model in prompt mode
may return "1,234.50" or "$1,234.50" as a string. Parse rather than assume:

    from core.normalize import parse_money, parse_date, is_blank

    amount = parse_money(row.get("amount"))     # float or None, handles $ , ( )
    when   = parse_date(record.get("due_date")) # date or None, several formats
    if is_blank(value): ...                     # None, "", whitespace, empty list


RETURN VALUE
------------
The number of things changed. It is reported on the prediction, in run.json, and in
the run summary, so a score can always be split between what the model did and what
cleanup did. Returning 0 when you changed something hides exactly that.
"""
from core.normalize import is_blank, parse_money  # noqa: F401  (referenced in docs)
from core.rules import RULES  # noqa: F401  (the decorator a real rule uses)


# Uncomment the decorator to make this live, and import the module in __init__.py.
#
# @RULES.register(
#     "drop_zero_quantity_rows",
#     applies_to=("invoice", "purchase_order"),
#     help="Remove line items billing a quantity of zero for zero.",
# )
def drop_zero_quantity_rows(record: dict) -> int:
    """Remove line items that bill nothing at all.

    A row with quantity 0 and amount 0 carries no charge and no information; it is
    ruled-table padding the model transcribed. Both must be zero -- a zero-amount row
    with a real quantity is a genuine free-of-charge line, and a zero-quantity row with
    an amount is a flat fee. Either is meaningful and is kept.
    """
    removed = 0

    def clean(container: dict) -> None:
        nonlocal removed
        rows = container.get("line_items")
        if not isinstance(rows, list):
            return
        kept = []
        for row in rows:
            if not isinstance(row, dict):
                kept.append(row)
                continue
            quantity = parse_money(row.get("quantity"))
            amount = parse_money(row.get("amount"))
            if quantity == 0 and amount == 0:
                removed += 1
                continue
            kept.append(row)
        container["line_items"] = kept

    clean(record)
    # Multi-bill invoices nest their line items one level down.
    for section in record.get("sections") or []:
        if isinstance(section, dict):
            clean(section)
    return removed
