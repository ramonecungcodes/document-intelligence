"""Calibration, and the properties the floor decision rests on.

Two of these guard claims made in prose elsewhere in the repo. The random-abstention
baseline is quoted in the module docstring as needing no sampling because the expected
accuracy of a uniform subset is the accuracy of the whole; that is an assertion about
arithmetic and is checked here. And a floor is only meaningful if the documents it
declines are still counted -- the failure mode is silent and flattering, so it gets a
test rather than a comment.
"""
import pytest

from classify.base import Classification
from eval.calibration import CalibrationScore, render
from route import signals


def score_of(pairs, profile="clean"):
    """(confidence, correct) pairs -> a scored set."""
    score = CalibrationScore()
    for confidence, correct in pairs:
        score.add(confidence, correct, profile)
    return score


class TestTheBaseline:
    def test_random_abstention_leaves_accuracy_where_it_started(self):
        """Every row's baseline is the overall accuracy, at any coverage.

        This is the comparison the whole table is read against, so if it ever drifts
        into being computed per-row from the answered subset it would silently become
        the same number as the accuracy column and every floor would look worthless.
        """
        score = score_of([(0.9, True), (0.8, True), (0.7, False), (0.6, True)])
        rows = score.curve()
        assert {row["baseline_accuracy"] for row in rows} == {0.75}

    def test_confidence_that_carries_nothing_matches_its_baseline(self):
        """Errors spread evenly through the confidence range buy nothing by declining.

        The point of the baseline column: this model's accuracy column sits on top of
        its random column, and a floor here is pure lost coverage.
        """
        score = score_of([(0.95, True), (0.85, False), (0.75, True), (0.65, False),
                          (0.55, True), (0.45, False), (0.35, True), (0.25, False)])
        for row in score.curve():
            if row["answered"] >= 4:
                assert abs(row["accuracy"] - row["baseline_accuracy"]) <= 0.15


class TestTheOperatingPoint:
    def test_it_takes_the_most_permissive_floor_that_clears_the_bar(self):
        """Among thresholds that hold the target, the useful one declines fewest.

        A floor exists to automate as much as possible at a stated error rate, so
        picking the *safest* qualifying threshold instead of the most permissive one
        would answer a question nobody asked and cost coverage for nothing.
        """
        score = score_of([(0.99, True), (0.95, True), (0.90, True), (0.85, True),
                          (0.50, False), (0.40, False)])
        point = score.operating_point(1.0)
        assert point["threshold"] == pytest.approx(0.55)
        assert point["coverage"] == pytest.approx(4 / 6, abs=1e-4)

    def test_a_model_that_cannot_reach_the_target_says_so(self):
        """None, not the best available. A floor that does not reach the standard is
        not a floor that nearly reaches it -- reporting the nearest one invites it to
        be shipped as though it qualified."""
        score = score_of([(0.99, False), (0.98, True), (0.97, False), (0.96, True)])
        assert score.operating_point(0.99) is None


class TestDeclinedDocumentsAreStillCounted:
    def test_the_withheld_answer_survives_abstention(self):
        """A classifier that declines still records what it was going to say.

        Without this the coverage curve can only be drawn above whatever floor was in
        force during the run, which makes the question 'is this floor set right?'
        unanswerable from the artifacts -- and answering it is the whole of Phase 5.
        """
        declined = Classification(doc_type="", withheld="purchase_order",
                                  confidence=0.42, engine="dit")
        assert declined.abstained
        row = signals.from_classification(declined)
        assert row["withheld"] == "purchase_order"
        assert row["abstained"] is True
        assert "doc_type" not in row

    def test_scoring_only_the_answered_reports_every_floor_as_free(self):
        """The failure this guards, stated as the arithmetic it produces.

        Six documents, the two wrong ones sitting under a 0.5 floor. Counting all six
        says the floor costs a third of the corpus to remove both errors. Counting only
        the four above it says the model is perfect and the floor is free.
        """
        everything = score_of([(0.9, True), (0.8, True), (0.7, True), (0.6, True),
                               (0.4, False), (0.3, False)])
        at_half = [r for r in everything.curve() if r["threshold"] == 0.5][0]
        assert at_half["coverage"] == pytest.approx(4 / 6, abs=1e-4)
        assert at_half["errors_caught"] == 2

        answered_only = score_of([(0.9, True), (0.8, True), (0.7, True), (0.6, True)])
        assert answered_only.curve()[0]["coverage"] == 1.0
        assert answered_only.operating_point(0.99)["coverage"] == 1.0


class TestReliability:
    def test_empty_bins_are_kept(self):
        """A model that never expresses middling confidence is saying something, and a
        table that omits the gap reads as a smooth curve instead."""
        score = score_of([(0.95, True), (0.05, False)])
        rows = score.bins()
        assert len(rows) == 10
        assert [r["n"] for r in rows] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    def test_the_gap_is_signed(self):
        """Over- and underconfidence have the same ECE and opposite consequences: only
        one of them routes errors past a floor."""
        over = score_of([(0.9, False)] * 5 + [(0.9, True)] * 5)
        assert over.to_dict()["mean_gap"] == pytest.approx(0.4)
        under = score_of([(0.6, True)] * 10)
        assert under.to_dict()["mean_gap"] == pytest.approx(-0.4)

    def test_a_perfectly_calibrated_model_has_no_error(self):
        score = score_of([(0.5, True)] * 5 + [(0.5, False)] * 5)
        assert score.expected_calibration_error() == pytest.approx(0.0)

    def test_thin_bins_do_not_become_the_headline(self):
        """MCE ignores bins under five. One document in a bin scores exactly 0 or 1 and
        produces a gap that is an artifact of the bin width."""
        score = score_of([(0.35, False)] + [(0.95, True)] * 20)
        assert score.max_calibration_error() < 0.1


class TestMissingConfidence:
    def test_classifiers_that_report_none_are_counted_apart(self):
        """A keyword classifier gives no confidence. Averaging over the ones that do
        while dropping the rest describes a different pipeline from the one that ran."""
        score = CalibrationScore()
        score.add(None, True)
        score.add(0.9, True)
        data = score.to_dict()
        assert data["unscored"] == 1
        assert data["documents"] == 1

    def test_nothing_to_score_renders_rather_than_raising(self):
        assert "nothing to score" in render(CalibrationScore())
