"""The comparison primitives.

These matter more than they look: a normaliser that is too strict understates every
accuracy number the project ever reports, and one that is too loose overstates them.
Both failures are silent, so they get pinned here.
"""
import datetime

import pytest

from core.normalize import (
    compare,
    digits_only,
    normalise_identifier,
    normalise_name,
    parse_bool,
    parse_date,
    parse_money,
)


class TestParsing:
    @pytest.mark.parametrize("value,expected", [
        ("2026-03-29", datetime.date(2026, 3, 29)),
        ("03/29/2026", datetime.date(2026, 3, 29)),      # how forms render
        ("Mar 29, 2026", datetime.date(2026, 3, 29)),    # how invoices render
        ("March 29, 2026", datetime.date(2026, 3, 29)),
        ("29 Mar 2026", datetime.date(2026, 3, 29)),
        ("not a date", None),
        ("", None),
        (None, None),
    ])
    def test_dates(self, value, expected):
        assert parse_date(value) == expected

    @pytest.mark.parametrize("value,expected", [
        ("$4,102.50", 4102.50),
        ("4102.5", 4102.50),
        ("INR 1,676,976.00", 1676976.00),
        ("(45.00)", -45.00),                              # accounting negative
        (-45.0, -45.0),
        ("", None),
        ("n/a", None),
    ])
    def test_money(self, value, expected):
        assert parse_money(value) == expected

    def test_bool_forms(self):
        assert parse_bool("Yes") is True
        assert parse_bool("checked") is True
        assert parse_bool("No") is False
        assert parse_bool("maybe") is None

    def test_identifier_ignores_case_and_separators(self):
        assert normalise_identifier("INV-20261000") == normalise_identifier("inv 20261000")

    def test_name_drops_legal_suffix(self):
        assert normalise_name("Acme, Inc.") == normalise_name("Acme Inc")
        assert normalise_name("Northwind Components LLC") == "northwind components"

    def test_digits_only(self):
        assert digits_only("868-43-9991") == "868439991"


class TestCompare:
    def test_both_blank_is_correct(self):
        """A defect that empties a field is correctly extracted as empty."""
        result = compare("identifier", "", "")
        assert result.match and result.exact

    def test_missing_is_not_a_match(self):
        assert not compare("identifier", "", "INV-1").match
        assert compare("identifier", "", "INV-1").note == "missing"

    def test_value_where_truth_is_blank_is_spurious(self):
        result = compare("identifier", "INV-1", "")
        assert not result.match
        assert result.note.startswith("predicted a value")

    def test_date_format_counts_as_a_match_but_not_exact(self):
        result = compare("date", "03/29/2026", "2026-03-29")
        assert result.match
        assert not result.exact
        assert result.normalised_only

    def test_money_tolerance(self):
        assert compare("money", "4102.505", 4102.50).match
        assert not compare("money", "4103.00", 4102.50).match

    def test_money_tolerance_is_configurable(self):
        assert compare("money", "4103.00", 4102.50, tolerance=1.0).match

    def test_name_fuzzy_accepts_punctuation_drift(self):
        assert compare("name", "Northwind Components, L.L.C.", "Northwind Components LLC").match

    def test_name_fuzzy_rejects_a_different_vendor(self):
        assert not compare("name", "Meridian Office Supply", "Northwind Components LLC").match

    def test_transposed_identifier_is_not_a_match(self):
        """4471 vs A471 is the ambiguity the review queue exists to resolve."""
        assert not compare("identifier", "A471", "4471").match

    def test_phone_ignores_formatting(self):
        assert compare("phone", "(964) 837-3315", "9648373315").match
