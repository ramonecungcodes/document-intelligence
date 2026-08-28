"""Document type definitions: what fields a type has, and how each is compared.

This is the first version of the type registry. Today only the scorer reads it, but
it is the same declaration the extractor will be prompted from, the validators will
check against, and the confidence signals will hang off -- so it is deliberately data
about documents rather than code about scoring.

Repeating groups matter more than they look. A multi-bill invoice is one document
carrying several separately-payable services, so `sections` is a repeating group whose
rows each contain their own repeating group of line items. Scoring has to match rows
before it can compare fields, and a missed row is a different failure from a wrong
field inside a matched one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Field:
    name: str
    kind: str = "text"          # see core.normalize.compare
    tolerance: Optional[float] = None
    threshold: Optional[float] = None
    key: bool = False           # identifies a row when matching a repeating group
    help: str = ""              # goes into the extraction schema as the description;
                                # required wherever a type shares a `kind` with a
                                # sibling field, or the model cannot tell them apart
    optional: bool = False      # the document may legitimately not carry this field
                                # at all, and saying so is part of the answer. See
                                # extract.schema for what it changes about the ask.
    weight: float = 1.0         # how much breaking this field costs, relative to a
                                # field of weight 1. Losing an invoice total and losing
                                # a vendor phone number are not the same event, and an
                                # unweighted accuracy prices them identically -- so a
                                # repair that trades one for the other reads as neutral
                                # when it is not. Weighted figures are reported
                                # ALONGSIDE unweighted ones, never instead: the
                                # unweighted number is the scientifically clean one and
                                # the weighted one is the operational one, and a single
                                # figure claiming to be both is neither.


@dataclass(frozen=True)
class Group:
    """A repeating group: line items, invoice sections, work history."""

    name: str
    fields: tuple = ()
    groups: tuple = ()          # nested groups, e.g. line items inside a section
    keys: tuple = ()            # fields that identify a row, if any
    help: str = ""              # what counts as a row -- and, as often matters more,
                                # what does not
    max_rows: int = 50          # hard ceiling in the schema; see extract.schema._array


@dataclass(frozen=True)
class DocType:
    """A document type, optionally with variants that differ in their field set.

    Forms are one type by pipeline stage but five by content: a W-9 and a loan
    application share almost nothing. Asking for the union means asking a model to
    return null for most of the schema, which costs tokens, time and accuracy. A
    variant narrows the request to the fields the document actually has.
    """

    name: str
    label_file: str             # labels/<file>.json
    fields: tuple = ()
    groups: tuple = ()
    variant_key: str = ""       # field whose value selects a variant, e.g. form_type
    variants: dict = field(default_factory=dict)

    def field_map(self):
        return {f.name: f for f in self.fields}

    def variant_of(self, record: dict) -> str:
        """Which variant a document belongs to, read from its own record."""
        return str(record.get(self.variant_key, "")) if self.variant_key else ""

    def graded_fields(self, variant: str = "") -> tuple:
        """The fields the extractor asks for and the scorer grades -- the same set.

        `fields_for` minus the variant key. That key is not extracted (it selected the
        schema, so asking for it is circular) and must not be graded either: the runner
        writes it from the corpus, so grading it would score a value we supplied
        ourselves. It was previously in both, wrong on all 160 forms; scoring it now
        would be equally wrong in the flattering direction.
        """
        return tuple(spec for spec in self.fields_for(variant)
                     if spec.name != self.variant_key)

    def fields_for(self, variant: str = "") -> tuple:
        """The fields to ask for: shared fields plus the variant's own.

        An unknown variant falls back to the union, which is wasteful but correct --
        better to over-ask than to silently drop fields the document really has.
        """
        if not self.variants:
            return self.fields
        specific = self.variants.get(variant)
        if specific is None:
            seen, merged = set(), list(self.fields)
            for group in self.variants.values():
                for spec in group:
                    if spec.name not in seen:
                        seen.add(spec.name)
                        merged.append(spec)
            return tuple(merged)
        return self.fields + tuple(specific)


LINE_ITEMS = Group(
    name="line_items",
    help=("One entry per billed item or service line. Summary rows are not line items: "
          "skip Subtotal, Tax, Total, Due, Service total and any similar roll-up, even "
          "when they appear in the same table as the items. Those figures are captured "
          "by the subtotal, tax and total fields."),
    fields=(
        Field("description", "text"),
        Field("quantity", "number", tolerance=0.001),
        Field("unit_price", "money"),
        Field("amount", "money"),
    ),
)

_MONEY_TOTALS = (
    Field("subtotal", "money", weight=2.0),
    Field("tax", "money", weight=2.0),
    Field("total", "money", weight=3.0),
)

INVOICE = DocType(
    name="invoice",
    label_file="invoices",
    fields=(
        Field("invoice_number", "identifier", weight=3.0),
        Field("invoice_date", "date", weight=2.0),
        Field("due_date", "date"),
        Field("po_number", "identifier", weight=2.0),
        Field("vendor_name", "name", weight=2.0),
        Field("bill_to", "name"),
        Field("terms", "enum"),
        Field("currency", "enum"),
    ) + _MONEY_TOTALS,
    groups=(LINE_ITEMS,),
)

PURCHASE_ORDER = DocType(
    name="purchase_order",
    label_file="purchase_orders",
    fields=(
        Field("po_number", "identifier", weight=2.0),
        Field("po_date", "date"),
        Field("delivery_date", "date"),
        Field("buyer", "name"),
        Field("vendor", "name"),
        Field("terms", "enum"),
        Field("currency", "enum"),
    ) + _MONEY_TOTALS,
    groups=(LINE_ITEMS,),
)

MULTI_BILL_INVOICE = DocType(
    name="multi_bill_invoice",
    label_file="multi_bill_invoices",
    fields=(
        Field("invoice_number", "identifier", weight=3.0),
        Field("invoice_date", "date", weight=2.0),
        Field("due_date", "date"),
        Field("vendor_name", "name", weight=2.0),
        Field("bill_to", "name"),
        Field("master_account", "identifier"),
        Field("terms", "enum"),
        Field("currency", "enum"),
        Field("section_count", "number", tolerance=0),
    ) + _MONEY_TOTALS,
    groups=(
        Group(
            name="sections",
            # account_number is what accounts payable routes the payment on, so it is
            # the row identity -- and the field whose loss makes a section unpayable.
            keys=("account_number", "service_code"),
            fields=(
                Field("service_type", "text",
                      help="Name of the service as printed, e.g. 'Water Service'."),
                Field("service_code", "identifier", key=True,
                      help="Short alphabetic code for the service, usually three "
                           "letters such as WTR, GAS, ELC, MOB. Not an account number."),
                Field("account_number", "identifier", key=True,
                      help="The account this service is billed to and paid against, "
                           "shown in the Account column, e.g. UTL-679707 or BTN-465511. "
                           "This is the number a payment is routed by."),
                # reference_label is deliberately not extracted. On the page the
                # word IS the label for the number beside it -- "METER M3947745" --
                # so asking for it as a value asks the model to transcribe furniture,
                # and it reliably returns the whole phrase instead. The corpus keeps
                # it as ground truth about the document; the extractor does not ask.
                Field("reference_number", "identifier",
                      help="The reference identifier itself, printed after its label, "
                           "e.g. M3947745 for a meter or MOB/76795/DS1 for a circuit. "
                           "Not the account number."),
                # Absence is handled by `optional`, not by this text. The two must not
                # both explain it: the previous version ended with "return null", which
                # contradicts a schema that asks for status 'absent' instead, and a
                # description arguing with its own schema is what made reference_number
                # wrong for three rounds. This says what the value is; the schema says
                # what to do when there isn't one.
                Field("service_location", "text", optional=True,
                      help="Street address of the site this service is delivered to. "
                           "Begin at the street number: a label such as SITE or "
                           "LOCATION printed in front of it is not part of the "
                           "address. A cost centre, a service type, a meter number or "
                           "the billing address is not a service location."),
                Field("cost_center", "text",
                      help="Two parts, both required: the CC- code and the department "
                           "name printed immediately after it. Copy the run of text "
                           "from the code up to the end of the department name. "
                           "'CC-2040 Operations' is correct. Returning the code alone, "
                           "'CC-2040', is the most common error and is wrong."),
                Field("service_period_start", "date",
                      help="First date of this service's billing period."),
                Field("service_period_end", "date",
                      help="Last date of this service's billing period."),
                Field("subtotal", "money", weight=2.0),
                Field("tax", "money", weight=2.0),
                Field("total", "money", weight=3.0),
            ),
            groups=(LINE_ITEMS,),
        ),
    ),
)

RESUME = DocType(
    name="resume",
    label_file="resumes",
    fields=(
        Field("name", "name"),
        Field("email", "email"),
        Field("phone", "phone"),
        Field("location", "text"),
        Field("target_role", "text"),
        Field("current_title", "text"),
        Field("years_experience", "number", tolerance=0),
    ),
    groups=(
        Group(
            name="work_history",
            keys=("company",),
            fields=(
                Field("company", "name", key=True),
                Field("title", "text"),
                Field("start_year", "number", tolerance=0),
                Field("end_year", "text"),
            ),
        ),
    ),
)

# Forms are one pipeline type but five different documents. Each variant declares only
# what its own paperwork carries; see DocType.fields_for.
_ONBOARDING = (
    Field("employee_name", "name"), Field("ssn", "ssn", weight=3.0),
    Field("date_of_birth", "date"), Field("home_address", "text"),
    Field("personal_email", "email"), Field("phone", "phone"),
    Field("job_title", "text"), Field("department", "text"),
    Field("manager", "name"), Field("start_date", "date"),
    Field("employment_type", "enum"), Field("pay_rate", "money"),
    Field("pay_frequency", "enum"), Field("bank_name", "name"),
    Field("bank_routing", "account", weight=3.0), Field("bank_account", "account", weight=3.0),
    Field("w4_filing_status", "enum"), Field("allowances", "number", tolerance=0),
    Field("emergency_contact_name", "name"), Field("emergency_contact_phone", "phone"),
    Field("i9_verified", "bool"), Field("handbook_acknowledged", "bool"),
)

_CLAIM = (
    Field("claim_number", "identifier",
          help="The claim's own number, usually prefixed CLM."),
    Field("policy_number", "identifier",
          help="The insurance policy number, not the claim number."),
    Field("claimant_name", "name"), Field("claim_type", "enum"),
    Field("date_of_loss", "date",
          help="When the loss happened, not when it was reported."),
    Field("date_reported", "date",
          help="When the claim was reported, on or after the date of loss."),
    Field("incident_location", "text"), Field("loss_description", "text"),
    Field("claim_amount", "money"), Field("deductible", "money"),
    Field("adjuster_name", "name"), Field("status", "enum"),
    Field("contact_phone", "phone"),
)

_W9 = (
    Field("name", "name", help="Name as shown on the income tax return."),
    Field("business_name", "name", optional=True,
          help="Trade or business name, if different from the name above."),
    Field("tax_classification", "enum"), Field("address", "text"),
    Field("city_state_zip", "text"),
    Field("tin_type", "enum", help="Which identifier is given: SSN or EIN."),
    Field("ssn", "ssn", weight=3.0,
          help="Only if tin_type is SSN; a W-9 carries one, never both."),
    Field("ein", "ein", weight=3.0,
          help="Only if tin_type is EIN; a W-9 carries one, never both."),
    Field("requester", "name"),
)

_W4 = (
    Field("name", "name"), Field("ssn", "ssn", weight=3.0), Field("address", "text"),
    Field("filing_status", "enum"), Field("multiple_jobs", "bool"),
    Field("dependents_amount", "money"), Field("other_income", "money"),
    Field("deductions", "money"), Field("extra_withholding", "money"),
)

_LOAN = (
    Field("application_number", "identifier"), Field("applicant_name", "name"),
    Field("ssn", "ssn", weight=3.0), Field("date_of_birth", "date"), Field("address", "text"),
    Field("phone", "phone"), Field("email", "email"), Field("employer", "name"),
    Field("job_title", "text"), Field("years_employed", "number", tolerance=0),
    Field("annual_income", "money"), Field("loan_type", "enum"),
    Field("loan_amount", "money"), Field("loan_term_months", "number", tolerance=0),
    Field("loan_purpose", "text"), Field("down_payment", "money"),
    Field("monthly_debt", "money"), Field("credit_score", "number", tolerance=0),
    Field("co_applicant_name", "name", optional=True),
)

FORM = DocType(
    name="form",
    label_file="forms",
    # Shared by every variant: each of these forms ends in a signature block, and
    # whether it was signed is the single most common defect in the corpus. The block
    # was printed on the page and recorded nowhere, so no validator could reach it --
    # 89 injected defects invisible for want of two fields.
    fields=(
        Field("form_type", "enum"),
        Field("signature", "name",
              help="the name written on the signature line at the end of the form. "
                   "Empty when the form was left unsigned."),
        Field("sign_date", "date",
              help="the date written beside the signature, not any other date on the "
                   "form"),
    ),
    variant_key="form_type",
    variants={
        "onboarding": _ONBOARDING,
        "claim": _CLAIM,
        "w9": _W9,
        "w4": _W4,
        "loan": _LOAN,
    },
)

REGISTRY = {
    d.name: d for d in (INVOICE, PURCHASE_ORDER, MULTI_BILL_INVOICE, RESUME, FORM)
}
BY_LABEL_FILE = {d.label_file: d for d in REGISTRY.values()}

# Fields present in every label but not part of the extraction task.
NON_FIELD_KEYS = frozenset({
    "file", "doc_type", "layout", "irregularities", "source_file", "degradation",
})


def for_label_file(stem: str) -> Optional[DocType]:
    return BY_LABEL_FILE.get(stem)
