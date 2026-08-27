"""Values that are the wrong shape, out of range, or in the wrong order in time.

Three families that share a property worth naming: each one is checkable against the
document alone, with no reference to anything outside it. A routing number either has
nine digits or it does not; a credit score is inside 300-850 or it is not; a loss
reported before it happened is impossible whatever the rest of the file says.

That self-containment is why these belong in Phase 4 and not later. A rule needing an
external fact -- is this a real routing number, does this vendor exist -- is a lookup,
and a lookup that fails is ambiguous in the same way an extraction failure is: the
document may be fine and the source may be stale. These rules cannot fail that way.

Every one of them declines when the field is absent. A missing SSN is a job for
`required`, and having two rules fire on one absence would report a single defect
twice and make the precision number a statement about rule overlap.
"""
from __future__ import annotations

import datetime
import re

from validate.base import Finding, Validator, register

SSN = re.compile(r"^\d{3}-?\d{2}-?\d{4}$")
ROUTING = re.compile(r"^\d{9}$")
# Placeholders that are the right shape and still not a number anybody has.
SSN_PLACEHOLDERS = {"000-00-0000", "111-11-1111", "123-45-6789", "999-99-9999"}


def _text(value) -> str:
    return str(value).strip() if value is not None else ""


def _number(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _date(value):
    text = _text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


@register("format")
class Format(Validator):
    """Identifiers that have to be the right shape."""

    applies_to = ("form",)

    def check(self, record: dict, doctype, variant: str = "") -> list:
        out = []
        ssn = _text(record.get("ssn"))
        if ssn:
            if not SSN.match(ssn) or ssn in SSN_PLACEHOLDERS:
                out.append(Finding(
                    code="invalid_ssn_format", field="ssn",
                    message="social security number is not a valid one",
                    actual=ssn))
        routing = _text(record.get("bank_routing"))
        if routing and not ROUTING.match(routing):
            out.append(Finding(
                code="invalid_routing_number", field="bank_routing",
                message="routing number is not nine digits", actual=routing))
        return out


@register("range")
class Range(Validator):
    """Numbers that have to sit inside a band, or on the right side of zero."""

    applies_to = ("form",)

    def check(self, record: dict, doctype, variant: str = "") -> list:
        out = []
        score = _number(record.get("credit_score"))
        if score is not None and not (300 <= score <= 850):
            out.append(Finding(
                code="credit_score_out_of_range", field="credit_score",
                message="credit score is outside 300-850",
                expected="300-850", actual=f"{score:.0f}"))

        income = _number(record.get("annual_income"))
        if income is not None and income < 0:
            out.append(Finding(code="negative_income", field="annual_income",
                               message="annual income is negative",
                               actual=f"{income:.2f}"))

        claim = _number(record.get("claim_amount"))
        if claim is not None and claim < 0:
            out.append(Finding(code="negative_claim_amount", field="claim_amount",
                               message="claim amount is negative",
                               actual=f"{claim:.2f}"))

        down = _number(record.get("down_payment"))
        loan = _number(record.get("loan_amount"))
        if None not in (down, loan) and down > loan:
            out.append(Finding(
                code="down_payment_exceeds_loan", field="down_payment",
                message=f"down payment {down:.2f} is larger than the loan "
                        f"{loan:.2f}",
                expected=f"<= {loan:.2f}", actual=f"{down:.2f}"))
        return out


@register("temporal")
class Temporal(Validator):
    """Dates that have to run forwards, and periods that must not overlap."""

    def check(self, record: dict, doctype, variant: str = "") -> list:
        out = []

        loss = _date(record.get("date_of_loss"))
        reported = _date(record.get("date_reported"))
        if loss and reported and reported < loss:
            out.append(Finding(
                code="date_reported_before_loss", field="date_reported",
                message=f"reported {reported}, before the loss on {loss}",
                expected=f">= {loss}", actual=str(reported)))

        for index, job in enumerate(record.get("work_history") or []):
            if not isinstance(job, dict):
                continue
            # Years, not dates. The schema carries start_year and end_year, and a rule
            # written against start_date silently checked nothing at all -- it found no
            # field, compared no values, and reported a clean 0.000 that looked like
            # the defect being absent rather than the rule missing its target.
            start = _number(job.get("start_year"))
            end = _number(job.get("end_year"))
            if start and end and end < start:
                out.append(Finding(
                    code="impossible_employment_dates",
                    field=f"work_history[{index}].end_year",
                    message=f"role {index + 1} ends {end:.0f}, before it starts "
                            f"{start:.0f}",
                    expected=f">= {start:.0f}", actual=f"{end:.0f}"))
            if _text(job.get("start_year")) == "" and _text(job.get("end_year")) == "":
                out.append(Finding(
                    code="missing_employment_dates",
                    field=f"work_history[{index}].start_year",
                    message=f"role {index + 1} has no dates"))

        # Identical periods, not merely overlapping ones. Services on one bill are
        # normally billed over the same month, so a rule keyed on overlap fires on
        # clean documents -- it did, on 40 of them, which is how it was caught. What
        # the corpus injects is one section's period copied verbatim onto another, and
        # two services claiming the exact same window is the thing worth a person's
        # attention.
        seen = {}
        for index, section in enumerate(record.get("sections") or []):
            if not isinstance(section, dict):
                continue
            start = _text(section.get("service_period_start"))
            end = _text(section.get("service_period_end"))
            if not start or not end:
                continue
            if (start, end) in seen:
                out.append(Finding(
                    code="overlapping_service_periods",
                    field=f"sections[{index}].service_period_start",
                    message=f"services {seen[(start, end)] + 1} and {index + 1} claim "
                            f"the identical period {start} to {end}",
                    actual=f"{start}..{end}",
                    # A warning, not an error, and the corpus is what decided that.
                    # Two services legitimately share a billing window on 2 of 352
                    # clean documents, so this cannot gate anything -- but a bill
                    # charging one period twice is worth a person's eye. Severity is
                    # the difference between "stop" and "look".
                    severity="warning"))
            else:
                seen[(start, end)] = index
        return out
