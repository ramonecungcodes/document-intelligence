"""The splitter stage: the contract, the baselines, and how boundaries are scored.

The scoring gets more attention than the splitters, for the same reason it does in the
classifier tests. A splitter can be swapped; a metric that reads well while hiding a
merge cannot, and a merge is the expensive failure -- it hands the extractor two
unrelated documents and produces a record for a document that never existed.
"""
import pytest

from eval.splitting import SplitScore
from split.base import SPLITTERS, Split


class TestContract:
    def test_every_splitter_registers_and_declares_settings(self):
        assert {"single", "every_page", "by_type"} <= set(SPLITTERS)
        for name, cls in SPLITTERS.items():
            assert hasattr(cls, "SETTINGS"), f"{name} declares no settings"

    def test_spans_cover_every_page_exactly_once(self):
        spans = Split(boundaries=[2, 5], pages=7).spans()
        assert spans == [(0, 1), (2, 4), (5, 6)]
        assert sum(b - a + 1 for a, b in spans) == 7

    def test_a_file_with_no_boundaries_is_one_document(self):
        assert Split(boundaries=[], pages=3).count == 1


class TestScoring:
    def test_page_zero_is_never_credited(self):
        """It is a document start by definition. Counting it would hand a splitter
        that finds nothing a correct answer on every file."""
        s = SplitScore()
        s.add([], [], ["invoice"])
        assert s.hit == 0 and s.truth == 0

    def test_a_missed_boundary_is_a_merge_and_a_spurious_one_is_a_cut(self):
        s = SplitScore()
        s.add(predicted=[3], actual=[2])
        assert (s.merged, s.spurious, s.hit) == (1, 1, 0)

    def test_exact_counts_whole_files_not_boundaries(self):
        """Three-document bundles can post a healthy F1 while almost no file comes out
        right, and that is the difference between a demo and a pipeline."""
        s = SplitScore()
        s.add([1, 2], [1, 2])
        s.add([1], [1, 4])
        assert s.to_dict()["exact_files"] == 0.5
        assert s.recall == round(3 / 4, 4)

    def test_same_type_joins_are_scored_apart(self):
        """The join where one document follows another of its own type is the one a
        change-of-type splitter cannot see. Averaged in, it disappears."""
        s = SplitScore()
        s.add(predicted=[2], actual=[1, 2],
              doc_types=["invoice", "invoice", "resume"])
        d = s.to_dict()
        assert d["same_type_boundaries"] == 1
        assert d["same_type_recall"] == 0.0

    def test_f1_is_none_rather_than_zero_when_nothing_was_found(self):
        """`single` finds no boundaries, so its precision is undefined, not perfect."""
        s = SplitScore()
        s.add([], [1, 2])
        assert s.precision is None and s.f1 is None
        assert s.recall == 0.0
