"""The classifier stage: the contract, the baseline, and how it is scored.

The scoring gets more attention than the classifiers here, deliberately. A classifier
can be swapped; a metric that reads well while hiding the failure is what let a field
fabricate a co-applicant name on 25 documents unnoticed in Phase 1.
"""
import pytest

from classify.base import Classification, CLASSIFIERS
from classify.keyword import Keyword
from eval.classification import ClassificationScore


class TestRegistry:
    def test_both_classifiers_register(self):
        assert {"keyword", "llm"} <= set(CLASSIFIERS)

    def test_each_declares_its_settings(self):
        for name, cls in CLASSIFIERS.items():
            assert getattr(cls, "SETTINGS", None), f"{name} declares no settings"

    def test_the_llm_schema_offers_only_known_types_plus_unknown(self):
        """A free-text answer would need mapping back, and the mapping would quietly
        become the classifier."""
        from classify.llm import LLMClassifier
        from core.doctypes import REGISTRY
        allowed = LLMClassifier().schema()["properties"]["doc_type"]["enum"]
        assert set(allowed) == set(REGISTRY) | {"unknown"}


class TestKeywordBaseline:
    def classify(self, text, **kw):
        return Keyword(**kw).classify(text)

    def test_it_finds_an_invoice(self):
        assert self.classify("INVOICE\nInvoice Number: INV-4471\nBill To: Acme"
                             ).doc_type == "invoice"

    def test_it_finds_a_purchase_order(self):
        assert self.classify("PURCHASE ORDER\nP.O. Number: PO-7001\nShip To: Dock 4"
                             ).doc_type == "purchase_order"

    def test_it_finds_a_resume(self):
        assert self.classify("Jane Doe\nWork History\nSkills\nEducation"
                             ).doc_type == "resume"

    def test_multi_bill_beats_plain_invoice(self):
        """Both say 'Invoice'; only the per-service structure separates them, which is
        exactly the confusion a model has to beat the baseline on."""
        text = ("INVOICE\nMaster Account: 4471\nService Code: UTL-1\n"
                "Cost Centre: CC-2040\nService Period: Mar-Apr")
        assert self.classify(text).doc_type == "multi_bill_invoice"

    def test_it_abstains_on_an_unrecognisable_page(self):
        result = self.classify("lorem ipsum dolor sit amet")
        assert result.abstained
        assert result.doc_type == ""

    def test_it_abstains_on_a_tie_rather_than_flipping_a_coin(self):
        """A coin flip reported as a classification is worse than an honest blank."""
        result = self.classify("Skills\nBill To", abstain_on_tie=True)
        assert result.abstained
        result = self.classify("Skills\nBill To", abstain_on_tie=False)
        assert not result.abstained

    def test_confidence_is_a_ratio_not_a_count(self):
        """Three hits of five is a different confident from three of twenty."""
        result = self.classify("INVOICE\nBill To: x\nAmount Due: 5\nRemit To: y")
        assert 0 < result.confidence <= 1

    def test_it_records_what_convinced_it(self):
        assert self.classify("PURCHASE ORDER\nBuyer: x").evidence

    def test_empty_input_does_not_crash(self):
        assert self.classify("").abstained
        assert self.classify(None).abstained


class TestClassificationScoring:
    def score(self, pairs):
        s = ClassificationScore()
        for truth, predicted, *rest in pairs:
            s.add(truth, predicted, rest[0] if rest else "")
        return s

    def test_perfect_classification(self):
        d = self.score([("invoice", "invoice")] * 3).to_dict()
        assert d["accuracy"] == 1.0
        assert d["per_class"][0]["recall"] == 1.0

    def test_the_majority_baseline_is_always_reported(self):
        """160 forms against 40 resumes means guessing 'form' scores 0.8 for free."""
        d = self.score([("form", "form")] * 8 + [("resume", "resume")] * 2).to_dict()
        assert d["majority_baseline"] == 0.8

    def test_abstentions_count_against_accuracy(self):
        """A classifier cannot buy accuracy by declining the hard half."""
        d = self.score([("invoice", "invoice"), ("resume", "")]).to_dict()
        assert d["accuracy"] == 0.5
        assert d["precision_answered"] == 1.0
        assert d["abstained"] == 1

    def test_recall_and_precision_differ_when_a_type_is_over_predicted(self):
        d = self.score([("invoice", "invoice"),
                        ("multi_bill_invoice", "invoice")]).to_dict()
        rows = {r["type"]: r for r in d["per_class"]}
        assert rows["invoice"]["recall"] == 1.0
        assert rows["invoice"]["precision"] == 0.5
        assert rows["multi_bill_invoice"]["recall"] == 0.0

    def test_it_names_what_a_type_is_confused_with(self):
        """'0.6 recall' does not tell you where the errors went; this does."""
        d = self.score([("multi_bill_invoice", "invoice")] * 3 +
                       [("multi_bill_invoice", "multi_bill_invoice")] * 2).to_dict()
        row = {r["type"]: r for r in d["per_class"]}["multi_bill_invoice"]
        assert row["confused_with"] == "invoice"
        assert row["confused_n"] == 3

    def test_a_right_answer_in_second_place_is_counted_separately(self):
        """A threshold problem and a comprehension problem need different fixes."""
        d = self.score([("multi_bill_invoice", "invoice", "multi_bill_invoice")]).to_dict()
        assert d["right_answer_was_second"] == 1

    def test_the_matrix_survives_json(self):
        import json
        json.dumps(self.score([("invoice", "resume")]).to_dict())

    def test_an_empty_score_does_not_divide_by_zero(self):
        d = ClassificationScore().to_dict()
        assert d["accuracy"] is None and d["majority_baseline"] is None

    def test_render_names_the_baseline(self):
        from eval.classification import render
        out = render(self.score([("form", "form")] * 3 + [("resume", "resume")]))
        assert "majority baseline" in out
        assert "CLASSIFICATION" in out
