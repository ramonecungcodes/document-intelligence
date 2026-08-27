"""The normalizer stage: the seam, and the two composites built on it.

The composites are tested against fake engines rather than real OCR. That is
deliberate: the thing worth pinning is the routing -- when does a cascade escalate,
which result wins, what gets recorded about the decision -- and none of that should
depend on whether Tesseract is installed or how well it reads a fax. Engine quality is
measured on the corpus; engine *selection* is logic, and logic gets unit tests.
"""
import pytest

from normalize.base import Extracted, NORMALIZERS
from normalize.composite import Cascade, Consensus


# Long enough to clear the cascade's own min_chars floor; a fixture that trips the
# threshold under test measures the fixture rather than the code.
SAMPLE = "Invoice INV-4471 from Northwind Traders, total 1,240.00 due 2026-04-01"


def result(text=SAMPLE, confidence=0.9, layer="ocr",
           engine="fake", seconds=1.0):
    return Extracted(text=text, pages=1, layer=layer, engine=engine,
                     confidence=confidence, seconds=seconds)


class Fake:
    """An engine that returns what the test tells it to, and counts its calls."""

    def __init__(self, out):
        self.out = out
        self.calls = 0

    def read(self, path):
        self.calls += 1
        # A fresh copy each call: the composites mutate what they return.
        return Extracted(**{**self.out.__dict__})


def cascade(engines, **settings):
    built = Cascade(**settings)
    built._members = list(engines.items())
    return built


def consensus(engines, **settings):
    built = Consensus(**settings)
    built._members = list(engines.items())
    return built


class TestRegistry:
    def test_every_normalizer_is_registered(self):
        assert {"native", "tesseract", "doctr", "cascade", "consensus"} <= set(NORMALIZERS)

    def test_each_declares_its_settings(self):
        for name, cls in NORMALIZERS.items():
            assert getattr(cls, "SETTINGS", None), f"{name} declares no settings"

    def test_heavy_engines_are_importable_without_their_dependencies(self):
        """Listing a plugin this image cannot run beats pretending it does not exist."""
        from normalize.doctr import DocTR
        from normalize.tesseract import Tesseract
        assert "doctr" in DocTR(pretrained=False).describe()
        assert "tesseract" in Tesseract().describe()


class TestCascade:
    def test_a_confident_first_engine_stops_the_cascade(self):
        cheap, dear = Fake(result(confidence=0.95)), Fake(result(confidence=0.99))
        out = cascade({"cheap": cheap, "dear": dear}, escalate_below=0.80).read("x.pdf")
        assert cheap.calls == 1
        assert dear.calls == 0, "the expensive engine was paid for and not needed"
        assert out.engine == "cascade:fake"

    def test_low_confidence_escalates(self):
        cheap, dear = Fake(result(confidence=0.40)), Fake(result(confidence=0.95))
        out = cascade({"cheap": cheap, "dear": dear}, escalate_below=0.80).read("x.pdf")
        assert cheap.calls == 1 and dear.calls == 1
        assert out.confidence == 0.95

    def test_empty_text_escalates_however_confident(self):
        """An engine certain about nothing is still an engine that read nothing."""
        cheap = Fake(result(text="", confidence=1.0))
        dear = Fake(result(text="a real document with plenty of text", confidence=0.5))
        out = cascade({"cheap": cheap, "dear": dear}, escalate_below=0.80).read("x.pdf")
        assert dear.calls == 1
        assert out.text.startswith("a real document")

    def test_short_text_escalates(self):
        cheap = Fake(result(text="INV", confidence=0.99))
        dear = Fake(result(text="a much longer and more plausible page of text", confidence=0.7))
        out = cascade({"cheap": cheap, "dear": dear},
                      escalate_below=0.80, min_chars=40).read("x.pdf")
        assert dear.calls == 1
        assert len(out.text) > 40

    def test_a_native_text_layer_always_wins_immediately(self):
        """It is exact by construction and reports no confidence to compare against."""
        native = Fake(result(layer="native", confidence=None, engine="native"))
        dear = Fake(result(confidence=0.99))
        out = cascade({"native": native, "dear": dear}).read("x.pdf")
        assert dear.calls == 0
        assert out.engine == "cascade:native"

    def test_the_best_result_survives_even_if_none_pass(self):
        """Everything failed; return the least-bad rather than the last tried."""
        first = Fake(result(text="a reasonably long stretch of recognised text", confidence=0.6))
        second = Fake(result(text="worse", confidence=0.1))
        out = cascade({"a": first, "b": second}, escalate_below=0.95).read("x.pdf")
        assert second.calls == 1
        assert out.confidence == 0.6, "kept the worse of the two"

    def test_what_it_tried_is_recorded(self):
        """A score attributed to 'cascade' is unattributable; it must name the engine."""
        cheap, dear = Fake(result(confidence=0.4)), Fake(result(confidence=0.95))
        out = cascade({"cheap": cheap, "dear": dear}, escalate_below=0.80).read("x.pdf")
        assert len(out.tried) == 2
        assert "cheap=" in out.tried[0] and "dear=" in out.tried[1]
        assert out.provenance()["engine"] == "cascade:fake"

    def test_a_single_engine_is_refused(self):
        from normalize.composite import _members
        with pytest.raises(SystemExit, match="at least two"):
            _members(["tesseract"])

    def test_engines_may_be_given_as_a_string(self):
        from normalize.composite import _members
        assert len(_members("native,tesseract")) == 2


class TestConsensus:
    def test_every_engine_runs(self):
        a, b = Fake(result(confidence=0.9)), Fake(result(confidence=0.7))
        consensus({"a": a, "b": b}).read("x.pdf")
        assert a.calls == 1 and b.calls == 1, "consensus needs all of them, always"

    def test_the_most_confident_result_is_kept(self):
        a = Fake(result(text="lower confidence reading", confidence=0.5))
        b = Fake(result(text="higher confidence reading", confidence=0.9))
        out = consensus({"a": a, "b": b}).read("x.pdf")
        assert out.text == "higher confidence reading"

    def test_identical_readings_agree_completely(self):
        same = "the same page read the same way by both engines"
        out = consensus({"a": Fake(result(text=same, confidence=0.8)),
                         "b": Fake(result(text=same, confidence=0.7))}).read("x.pdf")
        assert out.agreement == pytest.approx(1.0)

    def test_divergent_readings_lower_the_agreement(self):
        out = consensus({"a": Fake(result(text="invoice total four thousand", confidence=0.8)),
                         "b": Fake(result(text="zzzz qqqq wwww eeee", confidence=0.7))}).read("x.pdf")
        assert out.agreement is not None and out.agreement < 0.5

    def test_whitespace_differences_do_not_count_as_disagreement(self):
        """Engines differ on line breaks long before they differ on a digit."""
        out = consensus({"a": Fake(result(text="Invoice  INV-4471\nTotal 1200.00")),
                         "b": Fake(result(text="Invoice INV-4471 Total 1200.00"))}).read("x.pdf")
        assert out.agreement > 0.95

    def test_agreement_is_none_when_only_one_engine_produced_text(self):
        out = consensus({"a": Fake(result(text="real text here", confidence=0.8)),
                         "b": Fake(result(text="", confidence=0.0))}).read("x.pdf")
        assert out.agreement is None
        assert out.text == "real text here"

    def test_agreement_reaches_the_report(self):
        same = "identical reading from both engines"
        out = consensus({"a": Fake(result(text=same)), "b": Fake(result(text=same))}).read("x.pdf")
        assert out.provenance()["agreement"] == pytest.approx(1.0)
        assert out.provenance()["engine"].startswith("consensus:")


class TestExtractedContract:
    def test_provenance_is_json_serialisable(self):
        import json
        json.dumps(result().provenance())

    def test_empty_detects_whitespace_only(self):
        assert Extracted(text="   \n ", pages=1, layer="none").empty
        assert not result().empty
