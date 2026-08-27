"""Money that has to add up.

This is the family worth building first, because it is the one a validator can be
certain about. Whether a signature is missing is a judgement about a page; whether
`quantity x unit_price` equals `amount` is arithmetic, and a rule that gets arithmetic
wrong has no excuse.

Tolerance is the only real design decision and it is not zero. Every one of these
figures was printed rounded to the cent, so a subtotal computed from unrounded line
amounts disagrees with the printed one by fractions routinely and legitimately. A rule
with no tolerance reports those as defects, and a defect rate that is mostly rounding
tells you nothing. A rule with too much tolerance stops noticing the injected errors.
The corpus settles it: the injected mismatches are large enough to be unmistakable, so
the tolerance is set to absorb rounding and nothing else.

Every check refuses to fire when the inputs are absent rather than treating a missing
number as zero. A document whose total the extractor failed to read is a document this
family has nothing to say about -- claiming `total_mismatch` there would be reporting
an extraction failure as a document defect, which is the confusion the whole stage is
arranged to avoid.
"""
from __future__ import annotations

from core.plugins import Setting
from validate.base import Finding, Validator, register


def _number(value):
    """A float, or None when the field is absent or unreadable as one."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _rows(record: dict, key: str = "line_items"):
    rows = record.get(key)
    return rows if isinstance(rows, list) else []


@register("arithmetic")
class Arithmetic(Validator):
    """Line items, subtotals, tax and totals, checked against each other."""

    SETTINGS = (
        Setting("tolerance", float, default=0.02,
                help="cents of slack, to absorb printed rounding and nothing more"),
    )
    applies_to = ("invoice", "purchase_order", "multi_bill_invoice")

    def __init__(self, tolerance: float = 0.02, **_):
        self.tolerance = tolerance

    def _close(self, a: float, b: float) -> bool:
        return abs(a - b) <= self.tolerance

    def _check_rows(self, rows, code: str, label: str) -> list:
        out = []
        for index, row in enumerate(rows):
            quantity = _number(row.get("quantity"))
            unit = _number(row.get("unit_price"))
            amount = _number(row.get("amount"))
            if amount is not None and amount < 0:
                out.append(Finding(
                    code="negative_line_amount", field=f"{label}[{index}].amount",
                    message=f"line {index + 1} has a negative amount",
                    actual=f"{amount:.2f}"))
            if None in (quantity, unit, amount):
                continue
            expected = round(quantity * unit, 2)
            if not self._close(expected, amount):
                out.append(Finding(
                    code=code, field=f"{label}[{index}].amount",
                    message=(f"line {index + 1}: {quantity} x {unit:.2f} "
                             f"is {expected:.2f}, not {amount:.2f}"),
                    expected=f"{expected:.2f}", actual=f"{amount:.2f}"))
        return out

    def _check_totals(self, record, rows, prefix="", codes=None,
                      rows_bad: bool = False) -> list:
        codes = codes or {}
        out = []
        subtotal = _number(record.get(f"{prefix}subtotal"))
        tax = _number(record.get(f"{prefix}tax"))
        total = _number(record.get(f"{prefix}total"))

        amounts = [_number(r.get("amount")) for r in rows]
        # A subtotal computed from line items that are themselves wrong is wrong by
        # construction, and reporting it is reporting the same defect twice. The corpus
        # tags the cause; a reviewer wants the cause too. Suppressing the consequence
        # is the difference between "line 3 is wrong" and three findings that all mean
        # line 3 is wrong.
        usable = rows and not rows_bad and all(a is not None for a in amounts)
        if usable and subtotal is not None:
            summed = round(sum(amounts), 2)
            if not self._close(summed, subtotal):
                out.append(Finding(
                    code=codes.get("subtotal", "subtotal_mismatch"),
                    field=f"{prefix}subtotal",
                    message=f"line items sum to {summed:.2f}, subtotal says "
                            f"{subtotal:.2f}",
                    expected=f"{summed:.2f}", actual=f"{subtotal:.2f}"))

        subtotal_bad = any(f.code in ("subtotal_mismatch", "section_total_mismatch")
                           for f in out)
        if None not in (subtotal, tax, total) and not (rows_bad or subtotal_bad):
            expected = round(subtotal + tax, 2)
            if not self._close(expected, total):
                out.append(Finding(
                    code=codes.get("total", "total_mismatch"), field=f"{prefix}total",
                    message=f"subtotal {subtotal:.2f} plus tax {tax:.2f} is "
                            f"{expected:.2f}, total says {total:.2f}",
                    expected=f"{expected:.2f}", actual=f"{total:.2f}"))
        # Negativity is reported by the caller, which knows whether this is an
        # invoice or one service inside one and therefore which code the corpus uses.
        return out

    def check(self, record: dict, doctype, variant: str = "") -> list:
        rows = _rows(record)
        out = self._check_rows(rows, "line_item_math_error", "line_items")
        rows_bad = any(f.code == "line_item_math_error" for f in out)
        out += self._check_totals(record, rows, rows_bad=rows_bad)

        sections = record.get("sections")
        if not isinstance(sections, list):
            return out

        # A multi-bill invoice is several bills in one document, so the arithmetic
        # nests: each service has to foot on its own, and the invoice has to be the sum
        # of the services. Checking only the outer total would miss a section that is
        # wrong by an amount another section is wrong by in the other direction.
        section_totals = []
        seen_accounts = {}
        for index, section in enumerate(sections):
            label = f"sections[{index}]"
            section_rows = _rows(section)
            section_rows_out = self._check_rows(
                section_rows, "section_line_item_math_error", f"{label}.line_items")
            out += section_rows_out
            out += self._check_totals(
                section, section_rows,
                codes={"subtotal": "section_total_mismatch",
                       "total": "section_total_mismatch"},
                rows_bad=any(f.code == "section_line_item_math_error"
                             for f in section_rows_out))
            total = _number(section.get("total"))
            if total is not None:
                section_totals.append(total)
                if total < 0:
                    out.append(Finding(
                        code="negative_section_total", field=f"{label}.total",
                        message=f"service {index + 1} has a negative total",
                        actual=f"{total:.2f}"))
            account = (section.get("account_number") or "").strip()
            if account:
                if account in seen_accounts:
                    out.append(Finding(
                        code="duplicate_section_account",
                        field=f"{label}.account_number",
                        message=f"account {account} also appears on service "
                                f"{seen_accounts[account] + 1}; the two cannot be "
                                f"paid separately",
                        actual=account))
                seen_accounts[account] = index

        declared = _number(record.get("section_count"))
        if declared is not None and int(declared) != len(sections):
            out.append(Finding(
                code="section_count_mismatch", field="section_count",
                message=f"header says {int(declared)} services, "
                        f"{len(sections)} are printed",
                expected=str(len(sections)), actual=str(int(declared))))

        subtotal = _number(record.get("subtotal"))
        if section_totals and len(section_totals) == len(sections) and subtotal is not None:
            tax = _number(record.get("tax")) or 0.0
            summed = round(sum(section_totals), 2)
            invoice_total = _number(record.get("total"))
            if invoice_total is not None and not self._close(summed, invoice_total):
                out.append(Finding(
                    code="invoice_total_not_sum_of_sections", field="total",
                    message=f"services total {summed:.2f}, invoice says "
                            f"{invoice_total:.2f}",
                    expected=f"{summed:.2f}", actual=f"{invoice_total:.2f}"))
        return out
