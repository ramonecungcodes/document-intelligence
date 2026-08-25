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
