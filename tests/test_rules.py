"""The rules registry, and the two rules that ship with it.

The registry's job is to make adding a rule cheap while keeping the dangerous kind
impossible to add by accident: scope is declared, changes are counted, and nothing is
applied silently.
"""
import pytest

from core.rules import RULES, Registry, Rule, RuleReport
from extract.rules.empty_rows import drop_empty_rows
from extract.rules.rollup import drop_rollup_rows


def section(rows, subtotal=100.0, tax=8.0, total=108.0):
    return {"subtotal": subtotal, "tax": tax, "total": total, "line_items": rows}


class TestRollupRule:
    def test_drops_a_subtotal_row_matching_the_subtotal(self):
        rec = section([{"description": "Widget", "amount": 100.0},
                       {"description": "Subtotal", "amount": 100.0}])
        assert drop_rollup_rows(rec) == 1
        assert [r["description"] for r in rec["line_items"]] == ["Widget"]

    def test_drops_service_total_and_due(self):
        rec = section([{"description": "Widget", "amount": 100.0},
                       {"description": "Service total", "amount": 108.0},
                       {"description": "Due", "amount": 108.0}])
        assert drop_rollup_rows(rec) == 2

    def test_drops_a_named_rollup_carrying_no_amount(self):
        assert drop_rollup_rows(section([{"description": "Subtotal", "amount": None}])) == 1

    def test_reaches_into_nested_sections(self):
        rec = {"sections": [section([{"description": "A", "amount": 100.0},
                                     {"description": "Tax", "amount": 8.0}])]}
        assert drop_rollup_rows(rec) == 1
        assert len(rec["sections"][0]["line_items"]) == 1

    def test_keeps_tax_billed_as_a_real_line_item(self):
        """The value test is the whole safety margin: a real invoice can bill tax."""
        rec = section([{"description": "Tax", "amount": 42.75}], tax=8.0)
        assert drop_rollup_rows(rec) == 0
        assert rec["line_items"][0]["description"] == "Tax"

    def test_keeps_an_item_whose_name_merely_contains_a_rollup_word(self):
        assert drop_rollup_rows(section([{"description": "Total station rental",
                                          "amount": 100.0}])) == 0

    def test_survives_a_record_with_no_line_items(self):
        assert drop_rollup_rows({"total": 1.0}) == 0


class TestEmptyRowsRule:
    def test_drops_a_wholly_blank_row(self):
        rec = section([{"description": "Widget", "amount": 100.0},
                       {"description": "", "amount": None, "quantity": None}])
        assert drop_empty_rows(rec) == 1

    def test_keeps_a_row_with_only_an_amount(self):
        assert drop_empty_rows(section([{"description": "", "amount": 12.0}])) == 0

    def test_cleans_work_history_too(self):
        rec = {"work_history": [{"company": "Acme"}, {"company": "", "title": ""}]}
        assert drop_empty_rows(rec) == 1


class TestRegistry:
    def test_both_shipped_rules_are_registered(self):
        assert {"drop_rollup_rows", "drop_empty_rows"} <= set(RULES.names())

    def test_rules_are_on_unless_switched_off(self):
        names = [r.name for r in RULES.enabled({})]
        assert "drop_rollup_rows" in names
        assert "drop_rollup_rows" not in [r.name for r in
                                          RULES.enabled({"drop_rollup_rows": False})]

    def test_an_unknown_rule_in_the_manifest_is_an_error(self):
        """Same principle as plugin settings: a setting nobody reads is the bug."""
        with pytest.raises(ValueError, match="no rule 'drop_rollup_row'"):
            RULES.enabled({"drop_rollup_row": True})

    def test_scope_limits_a_rule_to_its_declared_types(self):
        registry = Registry()
        registry.add(Rule(name="only_invoices", apply=lambda r: 1,
                          applies_to=("invoice",)))
        assert registry.apply({}, "invoice").total == 1
        assert registry.apply({}, "resume").total == 0

    def test_an_unscoped_rule_applies_everywhere(self):
        registry = Registry()
        registry.add(Rule(name="everywhere", apply=lambda r: 1))
        assert registry.apply({}, "anything").total == 1

    def test_duplicate_names_are_refused(self):
        registry = Registry()
        registry.add(Rule(name="x", apply=lambda r: 0))
        with pytest.raises(ValueError, match="duplicate"):
            registry.add(Rule(name="x", apply=lambda r: 0))

    def test_report_records_what_changed(self):
        report = RuleReport()
        report.record("a", 2)
        report.record("a", 1)
        report.record("b", 0)
        assert report.to_dict() == {"a": 3}
        assert report.total == 3

    def test_applying_reports_per_rule_counts(self):
        rec = {"sections": [section([{"description": "Widget", "amount": 100.0},
                                     {"description": "Subtotal", "amount": 100.0},
                                     {"description": "", "amount": None}])]}
        report = RULES.apply(rec, "multi_bill_invoice")
        assert report.applied.get("drop_rollup_rows") == 1
        assert report.applied.get("drop_empty_rows") == 1


class TestExampleTemplate:
    """The template is live code so it cannot rot, but registers nothing."""

    def test_it_imports_and_registers_nothing(self):
        from extract.rules import _example
        before = set(RULES.names())
        assert "drop_zero_quantity_rows" not in before
        assert _example.drop_zero_quantity_rows({}) == 0

    def test_the_example_rule_works_as_documented(self):
        from extract.rules._example import drop_zero_quantity_rows
        rec = {"line_items": [
            {"description": "Widget", "quantity": 2, "amount": 100.0},
            {"description": "Padding", "quantity": 0, "amount": 0},
        ]}
        assert drop_zero_quantity_rows(rec) == 1
        assert len(rec["line_items"]) == 1

    def test_it_keeps_meaningful_zeros(self):
        """Free-of-charge lines and flat fees are real; only both-zero is padding."""
        from extract.rules._example import drop_zero_quantity_rows
        rec = {"line_items": [
            {"description": "Free sample", "quantity": 3, "amount": 0},
            {"description": "Flat fee", "quantity": 0, "amount": 50.0},
        ]}
        assert drop_zero_quantity_rows(rec) == 0

    def test_it_reaches_nested_sections_as_documented(self):
        from extract.rules._example import drop_zero_quantity_rows
        rec = {"sections": [{"line_items": [{"description": "x", "quantity": 0, "amount": 0}]}]}
        assert drop_zero_quantity_rows(rec) == 1

    def test_it_survives_the_absences_the_contract_warns_about(self):
        """No line_items, None values, non-dict rows -- all documented as possible."""
        from extract.rules._example import drop_zero_quantity_rows
        assert drop_zero_quantity_rows({"file": "x.pdf", "doc_type": "invoice"}) == 0
        assert drop_zero_quantity_rows({"line_items": None}) == 0
        assert drop_zero_quantity_rows({"line_items": ["not a dict"]}) == 0
        assert drop_zero_quantity_rows({"line_items": [{"quantity": None}]}) == 0
