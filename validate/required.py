"""Fields a document is not allowed to be missing.

The largest defect family in the corpus and the one with the most room to be wrong,
because "missing" is only a defect relative to a claim about what should be there. The
type registry already carries that claim -- `optional=True` marks the fields a document
may legitimately not have, which Phase 1 added when three fields were being fabricated
on every document that lacked them.

So this rule reads the registry rather than a list of its own. A field that is optional
in the schema cannot be required by a validator, or the two halves of the system
disagree about the same document and the extractor is punished for obeying its
instructions.

Two things it deliberately does not do.

It does not fire on a field the extractor returned as absent when that field is
optional -- that is the typed decision working, not a defect.

And it does not require fields the corpus never injects an absence for. A rule that
demands something the ground truth never marks missing produces false positives on
every clean document, which is exactly what the self-test exists to catch, and the
honest fix is not to write the rule.
"""
from __future__ import annotations

from validate.base import Finding, Validator, register

# What the corpus calls it, and where it lives. Read off the generator's own defect
# injectors rather than guessed, because a rule keyed to the wrong field is a rule that
# reports a defect the document does not have.
REQUIRED = {
    "invoice": {"invoice_number": "missing_invoice_number",
                "bill_to": "missing_bill_to"},
    "purchase_order": {"po_number": "missing_po_number",
                       "vendor": "missing_vendor"},
    "resume": {"email": "missing_email", "phone": "missing_phone"},
}

# Forms differ by variant: a W-9 has no bank account to be missing, and demanding one
# would fire on every clean W-9 in the corpus.
REQUIRED_FORM = {
    # Every variant ends in a signature block, and whether it was signed is the largest
    # defect class in the corpus. These two were unreachable until the generator
    # started recording them and the registry started asking: the block was printed on
    # the page and written down nowhere, so 89 injected defects had neither ground
    # truth to compare against nor a schema field to arrive in.
    "": {"signature": "missing_signature", "sign_date": "missing_sign_date"},
    "onboarding": {"bank_account": "missing_bank_account"},
    "claim": {"adjuster_name": "missing_adjuster"},
}

# Fields whose emptiness the corpus records under a name of its own rather than as a
# plain absence -- an unmade choice reads differently from a lost value.
CHOICES = {
    "onboarding": {"w4_filing_status": "no_filing_status_selected"},
    "w4": {"filing_status": "no_filing_status_selected"},
    "w9": {"tax_classification": "no_tax_classification_selected",
           "tin_type": "no_tin_type_selected"},
}


def _absent(value) -> bool:
    """Empty in the way a document is empty, not in the way zero is a number."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _optional_names(doctype, variant: str) -> set:
    """Every field the schema says may legitimately be absent, groups included.

    Group fields were missed at first, and the omission was invisible because this
    rule happens not to require any of them by name. A validator that reads the
    schema for top-level fields and hardcodes its own opinion inside a repeating
    group is two rules wearing one name, and only one of them can be corrected by
    editing the registry.
    """
    names = {spec.name for spec in doctype.fields_for(variant)
             if getattr(spec, "optional", False)}
    for group in getattr(doctype, "groups", ()) or ():
        names |= {spec.name for spec in group.fields
                  if getattr(spec, "optional", False)}
    return names


@register("required")
class Required(Validator):
    """Fields the document type says must be there, and are not."""

    def check(self, record: dict, doctype, variant: str = "") -> list:
        optional = _optional_names(doctype, variant)
        wanted = dict(REQUIRED.get(doctype.name, {}))
        if doctype.name == "form":
            wanted.update(REQUIRED_FORM.get("", {}))
            wanted.update(REQUIRED_FORM.get(variant, {}))
            wanted.update(CHOICES.get(variant, {}))

        out = []
        for name, code in sorted(wanted.items()):
            if name in optional:
                # The schema says this may legitimately be absent. A validator that
                # overrode that would punish the extractor for answering correctly.
                continue
            if name not in record:
                # Never extracted at all, which is a different thing from extracted as
                # empty and is not this rule's business to call a defect.
                continue
            if _absent(record.get(name)):
                out.append(Finding(
                    code=code, field=name,
                    message=f"{name.replace('_', ' ')} is empty",
                    actual=""))

        # A W-9 carries its taxpayer number in one of two fields depending on which
        # kind it is, so "the TIN is missing" is not a check on any single field. Both
        # empty is the defect; either one filled is a complete answer.
        if (doctype.name == "form" and variant == "w9"
                and "ssn" in record and "ein" in record
                and _absent(record.get("ssn")) and _absent(record.get("ein"))):
            out.append(Finding(
                code="missing_tin", field="ssn",
                message="no taxpayer identification number, as SSN or EIN"))

        if doctype.name == "resume" and isinstance(record.get("skills"), list) \
                and not record["skills"]:
            out.append(Finding(code="no_skills_listed", field="skills",
                               message="no skills are listed"))

        for index, section in enumerate(record.get("sections") or []):
            if not isinstance(section, dict):
                continue
            if ("account_number" not in optional
                    and "account_number" in section
                    and _absent(section.get("account_number"))):
                out.append(Finding(
                    code="missing_section_account",
                    field=f"sections[{index}].account_number",
                    message=f"service {index + 1} has no account number, so it cannot "
                            f"be paid separately"))
            if ("service_period_start" not in optional
                    and "service_period_start" in section
                    and _absent(section.get("service_period_start"))
                    and _absent(section.get("service_period_end"))):
                out.append(Finding(
                    code="missing_section_period",
                    field=f"sections[{index}].service_period_start",
                    message=f"service {index + 1} has no service period"))
        return out
