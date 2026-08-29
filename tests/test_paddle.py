"""The PaddleOCR plugin, tested without PaddleOCR installed.

Every test here runs in the main environment, where the dependency is deliberately
absent -- the OCR stage writes a cache and nothing downstream imports the engine that
produced it, so the plugin has to be importable, registrable and configurable without
the package. If it were not, adding an engine would mean every developer installing
every engine.

What cannot be tested here is whether Paddle reads a page well. That is not a unit test,
it is the comparison through the extractor, and it belongs in a report.
"""
import pytest

from normalize.base import NORMALIZERS, build
from normalize.paddle import Paddle


class TestItLoadsWithoutTheDependency:
    def test_it_is_registered(self):
        assert "paddle" in NORMALIZERS

    def test_it_builds_and_describes_itself(self):
        engine = build(plugin="paddle")
        assert "PP-OCRv5" in engine.describe()

    def test_the_model_version_is_pinned_not_defaulted(self):
        """paddleocr 3.7.0 ships PP-OCRv6 as its default and downloads it silently --
        the first smoke run fetched PP-OCRv6_medium_det while the report would have
        said v5. A benchmark that cannot name the model it ran is not a benchmark."""
        import inspect

        assert Paddle().version == "PP-OCRv5"
        assert '"ocr_version": self.version' in inspect.getsource(Paddle.engine)
        assert Paddle(version="PP-OCRv4").describe().count("PP-OCRv4") == 1

    def test_provenance_names_what_ran(self):
        out = Paddle(version="PP-OCRv5", unwarp=False).provenance()
        assert out["version"] == "PP-OCRv5"
        assert out["unwarp"] is False
        assert out["dpi"] == "source"

    def test_the_missing_dependency_says_where_it_lives(self):
        """A plain ImportError would send someone to pip in the wrong environment.
        The OCR stage is installed separately on purpose and the message has to say so.
        """
        with pytest.raises(SystemExit) as caught:
            Paddle().engine()
        message = str(caught.value)
        assert "paddleocr" in message
        assert ".venv-paddle" in message


class TestGeometry:
    def test_a_quadrilateral_becomes_its_bounding_box(self):
        """Paddle returns four corners; every consumer here speaks rectangles. An
        engine whose boxes meant something different from the others' would score
        differently for reasons unrelated to reading the page."""
        polygon = [[10, 20], [90, 24], [88, 60], [8, 56]]
        assert Paddle._bbox(polygon) == (8.0, 20.0, 90.0, 60.0)

    def test_a_rotated_box_keeps_every_corner_inside(self):
        polygon = [[50, 10], [90, 50], [50, 90], [10, 50]]
        x0, y0, x1, y1 = Paddle._bbox(polygon)
        for x, y in polygon:
            assert x0 <= x <= x1 and y0 <= y <= y1


class TestSettings:
    def test_unwarping_is_on_by_default(self):
        """It is the specific reason to try this engine: the photo profile carries the
        corpus's largest geometric distortion, and docTR's 31-point win over Tesseract
        was on exactly that profile."""
        assert Paddle().unwarp is True
        assert "unwarp" in Paddle().describe()

    def test_it_can_be_turned_off_to_price_it_separately(self):
        engine = Paddle(unwarp=False, orient=False)
        assert "plain" in engine.describe()

    def test_dpi_defaults_to_the_page_so_engines_get_the_same_input(self):
        """Comparing engines at different resolutions compares resolutions."""
        assert Paddle().dpi == 0
        assert "source dpi" in Paddle().describe()

    def test_textline_orientation_is_off(self):
        """This corpus does not produce rotated lines within a page, and leaving it on
        costs time for nothing. Asserted because it is a default that would otherwise
        drift back on with a library upgrade."""
        engine = Paddle()
        assert engine.orient is True          # document-level, kept
        # The textline switch is set inside engine(); its absence from SETTINGS is the
        # point -- it is not a knob, it is a decision.
        assert not any(s.name == "textline_orientation" for s in Paddle.SETTINGS)


class TestItAnswersLikeEveryOtherNormalizer:
    def test_it_reports_the_engine_name_it_will_be_cached_under(self):
        """`cached --engine paddle` reads the tree this writes. A mismatch between the
        name in Extracted.engine and the plugin name would cache under one name and be
        read under another."""
        import inspect

        source = inspect.getsource(Paddle.read)
        assert 'engine="paddle"' in source

    def test_an_empty_document_is_reported_as_no_layer(self):
        import inspect

        source = inspect.getsource(Paddle.read)
        assert 'layer="none"' in source


class TestDetectionGranularity:
    """Paddle detects lines; docTR and Tesseract detect words.

    `Extracted.words` is a contract. Two engines filling it with different granularity
    breaks every consumer silently -- `words_per_page` is a routing signal measured at
    +0.107 lift and would mean something different per engine, and the LayoutLM
    features expect word-level tokens.
    """

    def test_a_line_becomes_words(self):
        out = Paddle._split("Northwind Components LLC", (0.0, 10.0, 120.0, 30.0),
                            0.9, 1)
        assert [w.text for w in out] == ["Northwind", "Components", "LLC"]

    def test_the_boxes_tile_the_line_left_to_right(self):
        out = Paddle._split("alpha beta gamma", (0.0, 0.0, 100.0, 10.0), 0.9, 1)
        assert out[0].x0 == 0.0
        for earlier, later in zip(out, out[1:]):
            assert earlier.x1 <= later.x0 + 1e-9
        assert out[-1].x1 <= 100.0 + 1e-6

    def test_every_word_keeps_the_line_confidence_and_page(self):
        """Paddle scores a detection, not a token. Inventing per-word confidence would
        be a number with nothing behind it."""
        out = Paddle._split("alpha beta", (0.0, 0.0, 50.0, 10.0), 0.77, 3)
        assert all(w.confidence == 0.77 and w.page == 3 for w in out)

    def test_a_single_token_keeps_the_whole_box(self):
        out = Paddle._split("INVOICE", (5.0, 1.0, 45.0, 11.0), 0.9, 1)
        assert len(out) == 1
        assert (out[0].x0, out[0].x1) == (5.0, 45.0)

    def test_empty_text_produces_nothing(self):
        assert Paddle._split("   ", (0.0, 0.0, 10.0, 10.0), 0.9, 1) == []

    def test_a_zero_width_box_does_not_divide_by_zero(self):
        out = Paddle._split("alpha beta", (7.0, 0.0, 7.0, 10.0), 0.9, 1)
        assert len(out) == 1 and out[0].text == "alpha beta"
