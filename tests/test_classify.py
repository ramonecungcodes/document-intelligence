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
        assert {"cascade", "dit", "keyword", "layout", "llm"} <= set(CLASSIFIERS)

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


class TestCascade:
    """The tie-breaker. Its whole value is being narrow, so that is what is pinned."""

    def build(self, **kw):
        from core import config as config_mod
        from classify.cascade import Cascade
        c = Cascade(**kw)
        c.bind(config_mod.load())
        return c

    def stub(self, doc_type, runner_up="", variant="", confidence=0.99):
        from classify.base import Classification

        class Stub:
            NEEDS_TEXT = False
            SETTINGS = ()

            def classify(self, text="", **_):
                return Classification(doc_type=doc_type, variant=variant,
                                      runner_up=runner_up, confidence=confidence)
        return Stub()

    def test_it_does_not_escalate_when_the_pair_is_not_in_play(self):
        c = self.build(escalate_below=0.0)
        c._primary = self.stub("resume", runner_up="form")
        c._secondary = self.stub("invoice")
        assert c.classify(path="x.pdf").doc_type == "resume"

    def test_it_escalates_when_the_top_two_are_the_confusable_pair(self):
        c = self.build(escalate_below=0.0)
        c._primary = self.stub("invoice", runner_up="purchase_order")
        c._secondary = self.stub("purchase_order")
        assert c.classify(path="x.pdf").doc_type == "purchase_order"

    def test_the_secondary_may_not_introduce_a_third_type(self):
        """It is consulted to settle two candidates, not to reopen the question."""
        c = self.build(escalate_below=0.0)
        c._primary = self.stub("invoice", runner_up="purchase_order")
        c._secondary = self.stub("resume")
        assert c.classify(path="x.pdf").doc_type == "invoice"

    def test_a_coarser_answer_does_not_cost_the_variant(self):
        """The keyword baseline has no notion of form variants. Letting its bare
        `form` win stripped four documents of the field set that selects 22 fields
        rather than 9."""
        c = self.build(escalate_below=1.0)
        c._primary = self.stub("form", runner_up="invoice", variant="w9",
                               confidence=0.4)
        c._secondary = self.stub("form")
        result = c.classify(path="x.pdf")
        assert (result.doc_type, result.variant) == ("form", "w9")

    def test_a_failed_escalation_keeps_the_first_answer(self):
        """The second opinion is an improvement, not a dependency."""
        class Broken:
            NEEDS_TEXT = False
            SETTINGS = ()

            def classify(self, *a, **k):
                raise RuntimeError("ocr fell over")
        c = self.build(escalate_below=0.0)
        c._primary = self.stub("invoice", runner_up="purchase_order")
        c._secondary = Broken()
        assert c.classify(path="x.pdf").doc_type == "invoice"


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
