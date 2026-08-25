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


@dataclass(frozen=True)
class Group:
    """A repeating group: line items, invoice sections, work history."""

    name: str
    fields: tuple = ()
    groups: tuple = ()          # nested groups, e.g. line items inside a section
    keys: tuple = ()            # fields that identify a row, if any


@dataclass(frozen=True)
class DocType:
    name: str
    label_file: str             # labels/<file>.json
    fields: tuple = ()
    groups: tuple = ()

    def field_map(self):
        return {f.name: f for f in self.fields}


LINE_ITEMS = Group(
    name="line_items",
    fields=(
        Field("description", "text"),
        Field("quantity", "number", tolerance=0.001),
        Field("unit_price", "money"),
        Field("amount", "money"),
    ),
)

_MONEY_TOTALS = (
    Field("subtotal", "money"),
    Field("tax", "money"),
    Field("total", "money"),
)

INVOICE = DocType(
    name="invoice",
    label_file="invoices",
    fields=(
        Field("invoice_number", "identifier"),
        Field("invoice_date", "date"),
        Field("due_date", "date"),
        Field("po_number", "identifier"),
        Field("vendor_name", "name"),
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
        Field("po_number", "identifier"),
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
        Field("invoice_number", "identifier"),
        Field("invoice_date", "date"),
        Field("due_date", "date"),
        Field("vendor_name", "name"),
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
                Field("service_type", "text"),
                Field("service_code", "identifier", key=True),
                Field("account_number", "identifier", key=True),
                Field("reference_number", "identifier"),
                Field("service_location", "text"),
                Field("cost_center", "identifier"),
                Field("service_period_start", "date"),
                Field("service_period_end", "date"),
                Field("subtotal", "money"),
                Field("tax", "money"),
                Field("total", "money"),
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

# Forms vary their fields by form_type, so the registry carries the union and the
# scorer only grades fields present in the ground truth for that document.
FORM = DocType(
    name="form",
    label_file="forms",
    fields=(
        Field("form_type", "enum"),
        # identity
        Field("employee_name", "name"), Field("claimant_name", "name"),
        Field("applicant_name", "name"), Field("name", "name"),
        Field("business_name", "name"),
        Field("date_of_birth", "date"), Field("start_date", "date"),
        # contact
        Field("home_address", "text"), Field("address", "text"),
        Field("city_state_zip", "text"), Field("personal_email", "email"),
        Field("email", "email"), Field("phone", "phone"),
        Field("contact_phone", "phone"), Field("emergency_contact_name", "name"),
        Field("emergency_contact_phone", "phone"),
        # government identifiers
        Field("ssn", "ssn"), Field("ein", "ein"),
        Field("tin_type", "enum"), Field("tax_classification", "enum"),
        Field("filing_status", "enum"), Field("w4_filing_status", "enum"),
        Field("requester", "name"),
        # employment
        Field("job_title", "text"), Field("department", "text"),
        Field("manager", "name"), Field("employment_type", "enum"),
        Field("pay_rate", "money"), Field("pay_frequency", "enum"),
        Field("employer", "name"), Field("years_employed", "number", tolerance=0),
        Field("annual_income", "money"),
        # banking
        Field("bank_name", "name"), Field("bank_routing", "account"),
        Field("bank_account", "account"),
        # claim
        Field("claim_number", "identifier"), Field("policy_number", "identifier"),
        Field("claim_type", "enum"), Field("date_of_loss", "date"),
        Field("date_reported", "date"), Field("incident_location", "text"),
        Field("loss_description", "text"), Field("claim_amount", "money"),
        Field("deductible", "money"), Field("adjuster_name", "name"),
        Field("status", "enum"),
        # loan
        Field("application_number", "identifier"), Field("loan_type", "enum"),
        Field("loan_amount", "money"), Field("loan_term_months", "number", tolerance=0),
        Field("loan_purpose", "text"), Field("down_payment", "money"),
        Field("monthly_debt", "money"), Field("credit_score", "number", tolerance=0),
        Field("co_applicant_name", "name"),
        # withholding
        Field("allowances", "number", tolerance=0),
        Field("dependents_amount", "money"), Field("other_income", "money"),
        Field("deductions", "money"), Field("extra_withholding", "money"),
        Field("multiple_jobs", "bool"), Field("i9_verified", "bool"),
        Field("handbook_acknowledged", "bool"),
    ),
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
