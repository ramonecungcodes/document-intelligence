"""Scoring, including the two baselines the whole harness rests on.

The samples committed under tools/document-generator/samples are used as fixtures, so
these run without building a corpus first.
"""
import copy
import json
import os

import pytest

from core.doctypes import MULTI_BILL_INVOICE
from eval.score import match_rows, score

SAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "document-generator", "samples",
)


def _sample(name):
    with open(os.path.join(SAMPLES, name + ".json"), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def corpus(tmp_path):
    """A miniature corpus: the committed samples, laid out the way the generator does."""
    labels = tmp_path / "labels"
    labels.mkdir()
    records = {
        "invoices": [_sample("invoice")],
        "multi_bill_invoices": [_sample("multi-bill-invoice"),
                                _sample("multi-bill-invoice.defective")],
        "resumes": [_sample("resume")],
        "forms": [_sample("form-w9")],
    }
    for stem, items in records.items():
        (labels / f"{stem}.json").write_text(json.dumps(items), encoding="utf-8")
    return str(tmp_path), records


def _all(records):
    return [r for items in records.values() for r in items]


class TestBaselines:
    def test_ground_truth_scores_exactly_one(self, corpus):
        """The self-test. If this drifts below 1.0 a normaliser is wrong and every
        number the project reports afterwards is quietly understated."""
        root, records = corpus
        report = score(root, _all(records))
        assert report.overall()["field_accuracy"] == 1.0
        assert report.overall()["field_exact"] == 1.0

    def test_empty_extractor_scores_zero_on_fields_that_have_a_value(self, corpus):
        """An extractor returning nothing still "agrees" about deliberately emptied
        fields, so raw accuracy is not zero. The non-blank number is the honest one,
        and it must be."""
        root, records = corpus
        stubs = [{"file": r["file"]} for r in _all(records)]
        overall = score(root, stubs).overall()
        assert overall["field_accuracy_nonblank"] == 0.0
        assert overall["blank_fields"] > 0
        assert overall["field_accuracy"] == pytest.approx(
            overall["blank_fields"] / overall["fields_graded"], abs=1e-4)

    def test_no_predictions_at_all_grades_nothing(self, corpus):
        root, records = corpus
        report = score(root, [])
        assert report.overall()["scored"] == 0
        assert any("no matching prediction" in w for w in report.warnings)


class TestGroups:
    def test_sections_are_matched_and_scored(self, corpus):
        root, records = corpus
        report = score(root, _all(records), only=["multi_bill_invoices"])
        mb = next(s for s in report.slices if s.name == "multi_bill_invoices")
        sections = mb.groups["sections"]
        assert sections.truth_rows == sections.matched_rows > 0
        assert "line_items" in sections.groups          # nested group scored too

    def test_a_dropped_section_costs_row_recall_not_field_accuracy(self, corpus):
        """Missing a whole billable service must not hide inside a field average."""
        root, records = corpus
        predictions = copy.deepcopy(_all(records))
        target = next(p for p in predictions if len(p.get("sections", [])) > 1)
        dropped = target["sections"].pop()
        report = score(root, predictions, only=["multi_bill_invoices"])
        sections = next(
            s for s in report.slices if s.name == "multi_bill_invoices"
        ).groups["sections"]
        assert sections.matched_rows == sections.truth_rows - 1
        assert sections.to_dict()["row_recall"] < 1.0
        assert dropped is not None

    def test_rows_match_on_account_number_despite_reordering(self):
        group = next(g for g in MULTI_BILL_INVOICE.groups if g.name == "sections")
        truth = [
            {"account_number": "UTL-1", "service_code": "WTR", "total": 10.0},
            {"account_number": "UTL-2", "service_code": "GAS", "total": 20.0},
        ]
        pairs = match_rows(list(reversed(copy.deepcopy(truth))), truth, group)
        assert len(pairs) == 2
        for predicted, actual in pairs:
            assert predicted["account_number"] == actual["account_number"]

    def test_unrelated_rows_are_not_paired(self):
        group = next(g for g in MULTI_BILL_INVOICE.groups if g.name == "sections")
        truth = [{"account_number": "UTL-1", "service_code": "WTR", "total": 10.0}]
        junk = [{"account_number": "ZZZ-9", "service_code": "XXX", "total": 999.0}]
        assert match_rows(junk, truth, group) == []


class TestSlices:
    def test_degradation_slice_defaults_to_none(self, corpus):
        root, records = corpus
        report = score(root, _all(records))
        names = {s.name for s in report.slices if s.dimension == "degradation"}
        assert names == {"none"}

    def test_scanned_sample_lands_in_its_profile_slice(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        scanned = _sample("multi-bill-invoice.scanned")
        (labels / "multi_bill_invoices.json").write_text(
            json.dumps([scanned]), encoding="utf-8")
        report = score(str(tmp_path), [scanned])
        profiles = {s.name for s in report.slices if s.dimension == "degradation"}
        assert profiles == {"medium"}


class TestDetection:
    def test_perfect_detection(self, corpus):
        root, records = corpus
        report = score(root, _all(records))
        assert report.detection is not None
        assert report.detection.to_dict()["recall"] == 1.0
        assert report.detection.to_dict()["false_positive_rate_on_clean"] == 0.0

    def test_flagging_a_clean_document_is_counted(self, corpus):
        root, records = corpus
        predictions = copy.deepcopy(_all(records))
        clean = next(p for p in predictions if not p.get("irregularities"))
        clean["irregularities"] = ["total_mismatch"]
        report = score(root, predictions)
        detection = report.detection.to_dict()
        assert detection["clean_documents_flagged"] == 1
        assert detection["false_positive_rate_on_clean"] > 0
        assert detection["per_tag"]["total_mismatch"]["fp"] == 1

    def test_detection_is_skipped_when_nothing_predicts_it(self, corpus):
        root, records = corpus
        stripped = [{k: v for k, v in r.items() if k != "irregularities"}
                    for r in copy.deepcopy(_all(records))]
        assert score(root, stripped).detection is None


class TestReport:
    def test_report_is_json_serialisable_and_versioned(self, corpus):
        root, records = corpus
        data = json.loads(score(root, _all(records)).to_json())
        assert data["report_version"] == "1"
        assert "provenance" in data and "cost" in data
        assert set(data["cost"]) == {"tokens_in", "tokens_out", "usd", "latency_s"}


class TestPresenceMetrics:
    """The abstention split, on hand-built cases with known answers.

    One accuracy figure charges the same point for a value that was missed and a value
    that was invented. Those are not equally bad -- a blank field is honest and gets
    looked at, a confident wrong one flows downstream -- and averaging them hid a field
    that fabricated a co-applicant name on all 25 loan applications that had none.
    These assertions pin the arithmetic that separates them.
    """

    def grade(self, pairs, kind="text", name="service_location"):
        """Grade (predicted, truth) pairs for one field and return its FieldScore."""
        from core.doctypes import Field
        from eval.score import score_fields
        bucket = {}
        spec = Field(name, kind)
        for predicted, truth in pairs:
            score_fields({name: predicted}, {name: truth}, [spec], bucket)
        return bucket[name]

    def test_all_present_and_correct(self):
        p = self.grade([("123 Maple Avenue", "123 Maple Avenue")] * 4).presence()
        assert p["n_present"] == 4 and p["n_absent"] == 0
        assert p["presence_accuracy"] == 1.0
        assert p["precision_populated"] == 1.0
        assert p["recall_present"] == 1.0
        assert p["false_positive_rate"] is None       # nothing was absent to get wrong

    def test_all_absent_and_correctly_abstained(self):
        """The ssn/ein case: the model knows there is nothing to give."""
        p = self.grade([(None, None)] * 4).presence()
        assert p["n_absent"] == 4 and p["n_present"] == 0
        assert p["presence_accuracy"] == 1.0
        assert p["false_positive_rate"] == 0.0
        assert p["precision_populated"] is None       # it never populated anything
        assert p["recall_present"] is None            # nothing to recall

    def test_all_absent_and_all_invented(self):
        """The service_location case: sixteen chances to abstain, zero taken."""
        p = self.grade([(f"CC-{i}000 Operations", None) for i in range(4)]).presence()
        assert p["n_absent"] == 4
        assert p["false_positive_rate"] == 1.0
        assert p["presence_accuracy"] == 0.0
        assert p["precision_populated"] == 0.0

    def test_missing_values_hurt_recall_not_the_false_positive_rate(self):
        """Missing and invented are opposite errors and must not share a number."""
        p = self.grade([(None, "123 Maple Avenue")] * 4).presence()
        assert p["recall_present"] == 0.0
        assert p["presence_accuracy"] == 0.0
        assert p["false_positive_rate"] is None       # no absent cases at all
        assert p["precision_populated"] is None

    def test_a_mixed_field(self):
        p = self.grade([
            ("123 Maple Avenue", "123 Maple Avenue"),   # present, correct
            (None, None),                                # absent, abstained
            ("CC-2040 Operations", None),                # absent, invented
            (None, "77 Oak Street"),                     # present, missed
        ]).presence()
        assert p["n_absent"] == 2 and p["n_present"] == 2
        assert p["false_positive_rate"] == 0.5          # one of two absent invented
        assert p["recall_present"] == 0.5               # one of two present recovered
        assert p["precision_populated"] == 0.5          # two populated, one right
        assert p["presence_accuracy"] == 0.5            # right on two of four

    def test_the_rates_never_divide_by_zero(self):
        from eval.report import FieldScore
        p = FieldScore(name="empty").presence()
        assert all(p[k] is None for k in
                   ("presence_accuracy", "precision_populated",
                    "recall_present", "false_positive_rate", "contamination_rate"))


class TestContaminationDetection:
    """An invented value copied verbatim from a neighbour is a distinct diagnosis.

    Fifteen of sixteen invented service locations were exactly a sibling field's value,
    and the count did not move on a model four times larger. That pattern is what says
    the fault is in the schema rather than the model, so it is worth counting on its
    own rather than folding into a general wrongness figure.
    """

    def grade(self, predicted, truth, name="service_location"):
        from core.doctypes import Field
        from eval.score import score_fields
        bucket = {}
        score_fields(predicted, truth, [Field(name, "text")], bucket)
        return bucket[name]

    def test_a_value_lifted_from_a_sibling_is_counted(self):
        score = self.grade(
            {"service_location": "CC-2040 Operations", "cost_center": "CC-2040 Operations"},
            {"service_location": None, "cost_center": "CC-2040 Operations"})
        assert score.spurious == 1
        assert score.contaminated == 1

    def test_an_invented_value_of_its_own_is_not_contamination(self):
        """co_applicant_name fabricates names outright; that is a different failure."""
        score = self.grade(
            {"service_location": "Somewhere Entirely Else", "cost_center": "CC-2040"},
            {"service_location": None, "cost_center": "CC-2040"})
        assert score.spurious == 1
        assert score.contaminated == 0

    def test_a_correct_value_matching_a_sibling_is_not_contamination(self):
        """Two fields may legitimately hold the same value; only invention counts."""
        score = self.grade(
            {"service_location": "77 Oak Street", "billing_address": "77 Oak Street"},
            {"service_location": "77 Oak Street", "billing_address": "77 Oak Street"})
        assert score.spurious == 0
        assert score.contaminated == 0

    def test_short_values_are_not_treated_as_copies(self):
        """Two fields both holding 'N/A' is a coincidence, not a leak."""
        score = self.grade({"service_location": "N/A", "cost_center": "N/A"},
                           {"service_location": None, "cost_center": "N/A"})
        assert score.contaminated == 0

    def test_non_scalar_siblings_are_ignored(self):
        score = self.grade(
            {"service_location": "CC-2040 Operations", "line_items": [{"a": 1}]},
            {"service_location": None, "line_items": []})
        assert score.spurious == 1
        assert score.contaminated == 0

    def test_case_and_spacing_do_not_hide_a_copy(self):
        score = self.grade(
            {"service_location": "  cc-2040 OPERATIONS ", "cost_center": "CC-2040 Operations"},
            {"service_location": None, "cost_center": "CC-2040 Operations"})
        assert score.contaminated == 1

    def test_the_rate_separates_copied_inventions_from_original_ones(self):
        """Found by mutating contamination_rate to count every spurious value.

        Nothing failed. The counter was asserted but the rate derived from it was not,
        so the number actually printed in the report was untested. Both failures are
        inventions; only one is a sibling leak, and the whole diagnostic value is in
        telling them apart.
        """
        from core.doctypes import Field
        from eval.score import score_fields
        bucket = {}
        spec = Field("service_location", "text")
        score_fields({"service_location": "CC-2040 Operations", "cost_center": "CC-2040 Operations"},
                     {"service_location": None, "cost_center": "CC-2040 Operations"},
                     [spec], bucket)
        score_fields({"service_location": "Somewhere Entirely Else", "cost_center": "CC-9"},
                     {"service_location": None, "cost_center": "CC-9"}, [spec], bucket)
        p = bucket["service_location"].presence()
        assert p["false_positive_rate"] == 1.0     # both absent cases were invented
        assert p["contamination_rate"] == 0.5      # but only one was lifted from a sibling


class TestFieldNoteRendering:
    """A fabricated value must be visible in the printed report, not just the JSON.

    `spurious` was counted from the first version and written into report.json on every
    run. It was never printed. The note column was an elif chain that could only say
    "missing" -- the harmless error -- and the group table had no note column at all.
    So a field inventing a co-applicant name on all 25 loan applications that had none
    rendered as a bare 0.375 with nothing beside it, and stayed invisible through a
    full corpus run. The measurement was right; the rendering threw it away.
    """

    def field(self, **over):
        base = {"field": "co_applicant_name", "kind": "name", "n": 40,
                "exact": 0.375, "accuracy": 0.375, "accuracy_nonblank": 0.375,
                "blank": 0, "recovered_by_normalisation": 0, "missing": 0,
                "spurious": 0, "notes": {}}
        base.update(over)
        return base

    def test_an_invented_value_is_named(self):
        from eval.cli import _field_note
        assert _field_note(self.field(spurious=25)) == "25 invented"

    def test_invented_is_not_masked_by_missing(self):
        """The original bug: an elif meant only one of these could ever show."""
        from eval.cli import _field_note
        note = _field_note(self.field(spurious=25, missing=3))
        assert "25 invented" in note
        assert "3 missing" in note

    def test_invented_is_not_masked_by_normalisation(self):
        from eval.cli import _field_note
        note = _field_note(self.field(spurious=46, recovered_by_normalisation=34,
                                      notes={"fuzzy": 34}))
        assert "46 invented" in note
        assert "by fuzzy" in note

    def test_a_clean_field_says_nothing(self):
        from eval.cli import _field_note
        assert _field_note(self.field()) == ""

    def test_group_fields_get_the_same_note(self):
        """service_location lives inside `sections`, and that table had no note column.

        Fixing the top-level table alone left this one silent, which is exactly the
        drift that having two copies of the logic invited.
        """
        from eval.cli import _render_group
        out = []
        _render_group(out, {
            "group": "sections", "truth_rows": 36, "predicted_rows": 36,
            "matched_rows": 36, "row_precision": 1.0, "row_recall": 1.0, "row_f1": 1.0,
            "fields": [self.field(field="service_location", spurious=46)],
            "groups": [],
        })
        rendered = "\n".join(out)
        assert "service_location" in rendered
        assert "46 invented" in rendered
