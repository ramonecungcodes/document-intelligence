"""Turning a document into model input: the coordinate conversion, and its invariants.

This stage is silent when it goes wrong, which is why it is tested at all. A box on the
wrong scale does not raise -- it trains the model on a page whose contents are all
crowded into the top-left corner, and the only symptom is a number that is worse than
it should be for no visible reason. The same is true of a words list that has drifted
out of step with its boxes: nothing fails, the model just learns that some other word
was in that position.
"""
import pytest

from classify.features import MAX_WORDS, normalize_boxes
from normalize.base import Word


def word(text="x", page=1, x0=0.0, y0=0.0, x1=10.0, y1=10.0):
    return Word(text=text, page=page, x0=x0, y0=y0, x1=x1, y1=y1)


class TestCoordinates:
    def test_a_box_is_scaled_to_the_thousand_grid(self):
        """LayoutLM's position embedding has one entry per integer 0-1000. Points are
        not that, and a box passed through unconverted is silently in the corner."""
        texts, boxes = normalize_boxes([word(x0=0, y0=0, x1=306, y1=396)], 612, 792)
        assert texts == ["x"]
        assert boxes == [[0, 0, 500, 500]]

    def test_a_box_beyond_the_page_is_clamped_not_dropped(self):
        """OCR on a bad scan puts boxes slightly off-page. The word is still real."""
        _texts, boxes = normalize_boxes([word(x0=-5, y0=0, x1=700, y1=396)], 612, 792)
        assert boxes == [[0, 0, 1000, 500]]

    def test_an_inverted_box_is_normalised_rather_than_believed(self):
        texts, boxes = normalize_boxes([word(x0=306, y0=396, x1=0, y1=0)], 612, 792)
        assert texts == ["x"]
        assert boxes == [[0, 0, 500, 500]]

    def test_a_page_with_no_size_yields_nothing(self):
        """Better empty than divided by zero into coordinates that mean nothing."""
        assert normalize_boxes([word()], 0, 792) == ([], [])


class TestWhatIsDropped:
    def test_a_zero_area_box_goes_and_takes_its_word_with_it(self):
        """The two lists are positional. Dropping one side would shift every word
        after it onto the wrong box, and nothing would raise."""
        texts, boxes = normalize_boxes(
            [word("keep", x1=100, y1=100), word("flat", y1=0.0), word("also", x1=200, y1=200)],
            612, 792)
        assert texts == ["keep", "also"]
        assert len(texts) == len(boxes)

    def test_only_the_first_page_is_read(self):
        """The image is page one, so the words must be too. A model shown page one's
        picture beside page three's words is being trained on a contradiction."""
        texts, _boxes = normalize_boxes(
            [word("first", page=1, x1=100, y1=100),
             word("second", page=2, x1=100, y1=100)], 612, 792)
        assert texts == ["first"]

    def test_whitespace_is_not_a_word(self):
        texts, _ = normalize_boxes([word("   ", x1=100, y1=100)], 612, 792)
        assert texts == []

    def test_it_stops_at_the_sequence_limit(self):
        """Past this the tokenizer truncates anyway; carrying them costs memory to
        build inputs that are thrown away."""
        many = [word(f"w{i}", x1=100, y1=100) for i in range(MAX_WORDS + 50)]
        texts, boxes = normalize_boxes(many, 612, 792)
        assert len(texts) == MAX_WORDS
        assert len(boxes) == MAX_WORDS


class TestRegistration:
    def test_every_page_reading_classifier_registers(self):
        from classify.base import CLASSIFIERS
        assert {"dit", "layout"} <= set(CLASSIFIERS)

    def test_only_dit_declines_to_read_text(self):
        """NEEDS_TEXT is what lets the runner skip OCR, which is the expensive stage.
        A text classifier that inherited False would be handed an empty string."""
        from classify.base import CLASSIFIERS
        reads = {n: getattr(c, "NEEDS_TEXT", True) for n, c in CLASSIFIERS.items()}
        assert reads == {"dit": False, "layout": True, "keyword": True, "llm": True}

    def test_the_defaults_name_the_checkpoints_that_were_measured(self):
        """Both plugins pointed at an unweighted model at one stage. The unweighted
        LayoutLM answered `form` whenever unsure and scored 0.821 on fax against
        0.893; a default is how that difference gets published by accident."""
        from classify.dit import DEFAULT_MODEL as dit_default
        from classify.layout import DEFAULT_MODEL as layout_default
        assert dit_default.endswith("dit-balanced")
        assert layout_default.endswith("layout-balanced")

    def test_dit_refuses_without_a_path(self):
        """It rasterises the document itself, so a path is the one thing it needs."""
        from classify.dit import DocumentImage
        with pytest.raises(SystemExit):
            DocumentImage().classify("INVOICE")

    def test_the_layout_classifier_registers_without_torch(self):
        """Importing the package must not drag in a gigabyte of CUDA. di-app has no
        torch and classifies with the LLM; only training needs the heavy stack."""
        import sys
        from classify.base import CLASSIFIERS
        assert "layout" in CLASSIFIERS
        assert "torch" not in sys.modules

    def test_it_refuses_a_bare_string(self):
        """Given only text it would have no boxes and no image -- it would be a very
        expensive way to be a worse text classifier. Better to say so."""
        from classify.layout import Layout
        with pytest.raises(SystemExit):
            Layout().classify("INVOICE")
