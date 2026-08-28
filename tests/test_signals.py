"""Signal scoring, and the three ways it could flatter itself.

Every test here guards a way this module could report a signal as useful when it is
not. That is the only failure mode worth defending against: a diagnostic that
under-reports gets ignored, and a diagnostic that over-reports gets wired into routing.
"""
import pytest

from eval.signals import _percentiles, _ranks, report, score_signal, spearman
from route.features import EXPECTED, NAMES, profile_of, reading


class TestRankCorrelation:
    def test_a_monotone_relationship_is_one(self):
        assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == 1.0
        assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == -1.0

    def test_it_is_rank_based_not_value_based(self):
        """The whole reason for ranks: one ruined page must not set the correlation.

        The same ordering with a wild outlier appended keeps a perfect rho. On raw
        values that outlier would dominate the covariance.
        """
        clean = spearman([1, 2, 3, 4], [1, 2, 3, 4])
        outlier = spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 100000])
        assert clean == outlier == 1.0

    def test_ties_share_a_rank(self):
        """Competition ranking would order twenty documents that all scored zero
        validator errors by their position in the input file, and a correlation
        computed on that measures the sort order rather than the signal."""
        assert _ranks([5, 5, 5]) == [2.0, 2.0, 2.0]
        assert _ranks([1, 2, 2, 3]) == [1.0, 2.5, 2.5, 4.0]

    def test_a_constant_signal_is_untested_not_uncorrelated(self):
        """None, not 0.0. Zero means 'measured, and unrelated'; a signal that never
        varies has not been measured at all, and reporting it as zero retires a
        question nobody asked."""
        assert spearman([1, 1, 1, 1], [0.2, 0.4, 0.6, 0.8]) is None
        assert spearman([1, 2, 3, 4], [0.5, 0.5, 0.5, 0.5]) is None

    def test_too_few_points_is_none(self):
        assert spearman([1, 2], [1, 2]) is None


class TestOrientationIsDeclaredNotFitted:
    def test_direction_comes_from_the_table(self):
        """Flipping each signal to whichever sign correlates better is how a noise
        variable becomes a finding: with ten signals one will point the right way by
        chance, and letting the data pick the direction conceals that it was picked.
        """
        rising = _percentiles([1.0, 2.0, 3.0], +1)
        falling = _percentiles([1.0, 2.0, 3.0], -1)
        assert rising[0] < rising[-1]
        assert falling[0] > falling[-1]

    def test_a_contradicted_expectation_is_reported_as_one(self):
        """A signal that runs backwards must surface as a surprise, not be quietly
        re-oriented into a success."""
        # blank_share is declared negative; give it a positive relationship instead.
        pairs = [(share, share, "clean", "invoices")
                 for share in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
        row = score_signal("blank_share", pairs)
        assert EXPECTED["blank_share"] == -1
        assert row["rho"] > 0
        assert row["direction_agrees"] is False

    def test_every_signal_declares_a_direction(self):
        """A signal added to NAMES without an entry here would be oriented by the
        `or 1` fallback, silently, and its direction check would read as '-'."""
        assert set(NAMES) == set(EXPECTED)


class TestLiftIsTheNumberThatMatters:
    def test_a_correlated_signal_that_buys_nothing_shows_no_lift(self):
        """The distinction the table exists to draw. A signal can order documents
        correctly and still leave the answered set no better, when the outcomes it
        orders are all much the same."""
        pairs = [(index, 0.9, "clean", "invoices") for index in range(20)]
        row = score_signal("ocr_confidence", pairs)
        assert row["lift"] == pytest.approx(0.0, abs=1e-9)

    def test_a_useful_signal_beats_the_random_baseline(self):
        """Bad documents at the bottom of the signal: routing them away lifts what
        remains above the corpus mean, which is what random abstention would give."""
        pairs = ([(i, 0.2, "clean", "invoices") for i in range(5)]
                 + [(10 + i, 1.0, "clean", "invoices") for i in range(15)])
        row = score_signal("ocr_confidence", pairs)
        assert row["baseline"] == pytest.approx(0.8, abs=1e-3)
        assert row["lift"] > 0.1

    def test_coverage_is_fixed_across_signals(self):
        """Each signal evaluated at its own best coverage would always look good. The
        table compares them at one stated coverage or it compares nothing."""
        pairs = [(i, i / 20, "clean", "invoices") for i in range(20)]
        assert score_signal("ocr_confidence", pairs, 0.5)["coverage"] == 0.5
        assert score_signal("ocr_confidence", pairs, 0.8)["coverage"] == 0.8

    def test_a_signal_nobody_recorded_is_missing_not_zero(self):
        pairs = [(None, 0.9, "clean", "invoices")] * 10
        row = score_signal("rules_applied", pairs)
        assert row["available"] == 0
        assert row["missing"] == 10
        assert row["lift"] is None


class TestFeatures:
    def test_the_profile_comes_off_the_filename(self):
        assert profile_of("forms/onboarding_5000__fax.pdf") == "fax"
        assert profile_of("forms/onboarding_5000.pdf") == "clean"

    def test_engine_disagreement_is_read_from_the_normalizer_trace(self):
        """The losing engine's opinion survives only in `tried`, and two readers
        disagreeing about how much text is on a page is a fact about the page."""
        record = {"_normalizer": {
            "layer": "ocr", "pages": 1, "words": 100, "confidence": 0.9,
            "tried": ["tesseract=33ch/conf0.78/1.1s", "doctr=953ch/conf0.92/5.8s"]}}
        out = reading(record)
        assert out["engine_agreement"] == pytest.approx(33 / 953, abs=1e-4)
        assert out["engine_confidence_spread"] == pytest.approx(0.14, abs=1e-4)
        assert out["words_per_page"] == 100.0

    def test_one_engine_gives_no_disagreement(self):
        """None, not 1.0. A single reader has not agreed with anything."""
        record = {"_normalizer": {"layer": "ocr", "pages": 1,
                                  "tried": ["doctr=953ch/conf0.92/5.8s"]}}
        assert reading(record)["engine_agreement"] is None

    def test_a_native_page_has_no_ocr_confidence(self):
        """None rather than zero. No page has a legibility of zero, and substituting
        one would tell a calibrator the cleanest documents were the least readable."""
        out = reading({"_normalizer": {"layer": "native", "engine": "native",
                                       "pages": 1}})
        assert out["ocr_confidence"] is None
        assert out["is_ocr"] == 0.0


class TestTheReport:
    def test_small_groups_are_left_out_of_the_slices(self):
        """A per-type row computed on three documents is noise wearing a table's
        clothes."""
        rows = ([{"signals": {"ocr_confidence": i / 10}, "outcome": i / 10,
                  "profile": "fax", "truth": "invoices"} for i in range(10)]
                + [{"signals": {"ocr_confidence": 0.5}, "outcome": 0.5,
                    "profile": "fax", "truth": "resumes"} for _ in range(3)])
        data = report(rows)
        assert "invoices" in data["by_truth"]
        assert "resumes" not in data["by_truth"]
