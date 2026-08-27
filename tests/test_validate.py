"""The validator stage: the rules, and the self-test that has to pass before them.

A validator runs on extracted output, so when it fires there are two explanations --
the document is defective, or the extractor misread a good document -- and nothing in
the firing tells them apart. The tests that matter here are the ones pinning the
behaviours that keep those separable.
"""
import pytest

from core import doctypes
from validate.base import VALIDATORS, build_all, run
from validate.arithmetic import Arithmetic
from eval.validation import ValidationScore


def invoice(**over):
    base = {"subtotal": 100.0, "tax": 7.0, "total": 107.0,
            "line_items": [{"description": "x", "quantity": 2,
                            "unit_price": 25.0, "amount": 50.0},
                           {"description": "y", "quantity": 1,
                            "unit_price": 50.0, "amount": 50.0}]}
    base.update(over)
    return base


def check(record):
    return {f.code for f in Arithmetic().check(record, doctypes.REGISTRY["invoice"])}


class TestArithmetic:
    def test_a_document_that_foots_produces_nothing(self):
        assert check(invoice()) == set()

    def test_rounding_is_not_a_defect(self):
        """Every figure is printed to the cent, so a subtotal recomputed from rounded
        amounts disagrees by fractions legitimately. A rule with no tolerance reports
        a defect rate that is mostly rounding."""
        assert check(invoice(subtotal=100.01)) == set()

    def test_a_wrong_line_is_caught(self):
        assert "line_item_math_error" in check(
            invoice(line_items=[{"description": "x", "quantity": 2,
                                 "unit_price": 25.0, "amount": 61.0}]))

    def test_a_consequence_is_not_reported_as_a_second_defect(self):
        """Perturbing one line amount makes the subtotal disagree too. Reporting both
        says one thing twice, and a reviewer wants the cause."""
        found = check(invoice(line_items=[{"description": "x", "quantity": 2,
                                           "unit_price": 25.0, "amount": 61.0},
                                          {"description": "y", "quantity": 1,
                                           "unit_price": 50.0, "amount": 50.0}]))
        assert found == {"line_item_math_error"}

    def test_a_missing_figure_is_not_a_defect(self):
        """A total the extractor failed to read is not a document that fails to add up.
        Claiming one would report an extraction failure as a defect, which is the
        confusion this whole stage is arranged to avoid."""
        assert check(invoice(total=None)) == set()
        assert check(invoice(total="")) == set()

    def test_a_negative_line_is_caught_on_its_own_code(self):
        assert "negative_line_amount" in check(
            invoice(line_items=[{"description": "x", "quantity": 2,
                                 "unit_price": 25.0, "amount": -50.0}]))

    def test_it_declines_types_it_was_not_written_for(self):
        """A rule that runs everywhere and returns nothing for most types looks exactly
        like a rule that is broken."""
        assert Arithmetic().handles("invoice")
        assert not Arithmetic().handles("resume")


class TestSelfTestGate:
    def test_a_clean_document_flagged_on_truth_is_the_rule_being_wrong(self):
        s = ValidationScore()
        s.add(found={"total_mismatch"}, injected=set(), is_clean=True)
        d = s.to_dict()
        assert d["clean_documents_flagged"] == 1
        assert d["false_alarm_rate"] == 1.0

    def test_document_recall_counts_a_sibling_code_as_caught(self):
        """A tampered tax and a tampered total both present as subtotal + tax != total
        and nothing on the page says which. Naming the sibling still routes the
        document to a person, which is what the stage is for."""
        s = ValidationScore()
        s.add(found={"total_mismatch"}, injected={"tax_miscalculated"})
        d = s.to_dict()
        assert d["document_recall"] == 1.0
        assert d["recall"] == 0.0

    def test_an_unchecked_defect_class_shows_as_a_zero_not_an_absence(self):
        """A suite that silently declines to check something reads exactly like a
        suite that checks it and finds nothing."""
        s = ValidationScore()
        s.add(found=set(), injected={"missing_signature"})
        codes = {r["code"]: r for r in s.to_dict()["per_code"]}
        assert codes["missing_signature"]["recall"] == 0.0


class TestRegistry:
    def test_validators_register_and_build(self):
        assert "arithmetic" in VALIDATORS
        assert [v.name for v in build_all()] == sorted(VALIDATORS)

    def test_run_records_what_was_skipped(self):
        report = run(build_all(), {"name": "x"}, doctypes.REGISTRY["resume"])
        assert "arithmetic" in " ".join(report.skipped)
        assert report.ok


class TestConstraints:
    def check(self, record, doc_type="form", variant=""):
        from validate.constraints import Format, Range, Temporal
        out = []
        for v in (Format(), Range(), Temporal()):
            if v.handles(doc_type):
                out += v.check(record, doctypes.REGISTRY[doc_type], variant)
        return {f.code for f in out}

    def test_a_valid_ssn_passes_and_a_placeholder_does_not(self):
        assert self.check({"ssn": "412-88-7301"}) == set()
        assert "invalid_ssn_format" in self.check({"ssn": "000-00-0000"})

    def test_an_absent_field_is_not_this_rule_s_business(self):
        """A missing SSN belongs to `required`. Two rules firing on one absence
        reports a single defect twice and makes precision a statement about overlap."""
        assert self.check({"ssn": ""}) == set()
        assert self.check({}) == set()

    def test_employment_dates_are_years_not_dates(self):
        """A rule written against start_date checked nothing at all: it found no
        field, compared no values, and reported a clean 0.000 that read exactly like
        the defect being absent."""
        found = self.check({"work_history": [{"start_year": 2020, "end_year": 2015}]},
                           doc_type="resume")
        assert "impossible_employment_dates" in found

    def test_services_may_share_a_billing_window(self):
        """Clean multi-bill invoices bill several services over the same month, so a
        rule keyed on overlap fires on documents that are fine -- it did, on 40."""
        record = {"sections": [
            {"service_period_start": "2026-03-13", "service_period_end": "2026-04-13"},
            {"service_period_start": "2026-03-14", "service_period_end": "2026-04-13"}]}
        assert self.check(record, doc_type="multi_bill_invoice") == set()

    def test_an_identical_window_is_a_warning_not_an_error(self):
        """It happens by chance on 2 of 352 clean documents, so it cannot gate -- but
        a bill charging one period twice is worth a person's eye."""
        from validate.constraints import Temporal
        record = {"sections": [
            {"service_period_start": "2026-03-13", "service_period_end": "2026-04-13"},
            {"service_period_start": "2026-03-13", "service_period_end": "2026-04-13"}]}
        found = Temporal().check(record, doctypes.REGISTRY["multi_bill_invoice"])
        assert [f.severity for f in found] == ["warning"]


class TestRequired:
    def check(self, record, doc_type="invoice", variant=""):
        from validate.required import Required
        return {f.code for f in Required().check(
            record, doctypes.REGISTRY[doc_type], variant)}

    def test_an_empty_required_field_is_a_defect(self):
        assert "missing_bill_to" in self.check({"bill_to": "", "invoice_number": "x"})

    def test_a_field_never_extracted_is_not_a_defect(self):
        """Absent from the record entirely means the extractor did not produce it,
        which is a different thing from the document not carrying it."""
        assert self.check({"invoice_number": "x"}) == set()

    def test_the_schema_decides_what_is_optional_not_this_rule(self):
        """Phase 1 marked fields a document may legitimately lack. A validator that
        overrode that would punish the extractor for answering correctly."""
        from validate.required import _optional_names
        assert "service_location" in _optional_names(
            doctypes.REGISTRY["multi_bill_invoice"], "")
