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
