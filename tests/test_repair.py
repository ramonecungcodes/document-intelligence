"""Repair scoring, and the two numbers that must never be quotable alone.

A repair loop is the first stage in this project that can improve its own score by
making documents worse, so its scorer is written adversarially: every test here is a way
a loop could look successful without helping.
"""
import pytest

from eval.repair import Outcome, RepairScore, by_slice, compare, render


def outcome(before, after, gates_before=1, gates_after=0, **kw):
    return Outcome(file=kw.pop("file", "invoices/a.pdf"), before=before, after=after,
                   gates_before=gates_before, gates_after=gates_after, **kw)


def arm(name, pairs, **kw):
    score = RepairScore(arm=name)
    for index, (before, after) in enumerate(pairs):
        score.add(outcome(before, after, file=f"invoices/{index}.pdf", **kw))
    return score


class TestSilencingTheCritics:
    def test_clearing_gates_is_not_success(self):
        """The trap this scorer exists for. Every gate cleared, every document worse:
        exactly what blanking the fields a rule reads would produce."""
        score = arm("reprompt", [(0.9, 0.4), (0.8, 0.3), (0.7, 0.2)])
        data = score.to_dict()
        assert data["gates_clear"] == 3
        assert data["gates_clear_rate"] == 1.0
        assert data["net_delta"] < 0
        assert data["damaged"] == 3
        assert data["improved"] == 0

    def test_the_report_names_it_when_damage_exceeds_repair(self):
        data = compare({"reprompt": arm(
            "reprompt", [(0.9, 0.4), (0.9, 0.4), (0.9, 0.95)])})
        assert "damaged more documents than it improved" in render(data)

    def test_gates_and_accuracy_are_separate_columns(self):
        """A loop can be right about the document and still trip a rule, and can
        satisfy every rule while being wrong. Neither may stand in for the other."""
        score = RepairScore()
        score.add(outcome(0.5, 0.9, gates_before=2, gates_after=2))
        data = score.to_dict()
        assert data["improved"] == 1
        assert data["gates_clear"] == 0


class TestBothDirectionsAreReported:
    def test_improvement_and_damage_are_both_counted(self):
        score = arm("reprompt", [(0.5, 0.9), (0.9, 0.5), (0.7, 0.7)])
        data = score.to_dict()
        assert (data["improved"], data["damaged"], data["unchanged"]) == (1, 1, 1)
        assert data["net_delta"] == pytest.approx(0.0, abs=1e-9)

    def test_net_delta_is_the_sum_of_help_and_harm(self):
        """A loop is one decision -- run it or do not -- so what it is worth is the
        net. Sixty improved and fifty damaged is not sixty improved."""
        score = arm("reprompt", [(0.4, 0.9)] * 6 + [(0.9, 0.4)] * 5)
        data = score.to_dict()
        assert data["improved"] == 6 and data["damaged"] == 5
        assert data["net_delta"] > 0
        assert data["net_delta"] < 0.1        # nothing like "six documents better"

    def test_noise_is_not_a_repair(self):
        """Two answers to a sampled model differ slightly every time. A change at that
        scale is not a document changing."""
        score = arm("reprompt", [(0.80000, 0.80005), (0.7, 0.69995)])
        data = score.to_dict()
        assert data["improved"] == 0 and data["damaged"] == 0
        assert data["unchanged"] == 2


class TestTheBlindBaseline:
    def test_guided_repair_is_reported_against_a_plain_re_run(self):
        """The extractor is sampled, so a second request improves some documents by
        luck. Any repair inherits that for free, and the only question is whether the
        feedback adds anything to it."""
        data = compare({
            "rerun": arm("rerun", [(0.5, 0.6), (0.5, 0.6)]),
            "reprompt": arm("reprompt", [(0.5, 0.8), (0.5, 0.8)]),
        })
        assert data["baseline"] == "rerun"
        assert data["arms"]["rerun"]["over_rerun"] is None
        assert data["arms"]["reprompt"]["over_rerun"] == pytest.approx(0.2, abs=1e-4)

    def test_feedback_worth_nothing_is_said_in_words(self):
        """Equal to the baseline is a real result and has to read as one, not as a
        positive net_delta with a quiet footnote."""
        data = compare({
            "rerun": arm("rerun", [(0.5, 0.7), (0.5, 0.7)]),
            "reprompt": arm("reprompt", [(0.5, 0.7), (0.5, 0.7)]),
        })
        assert data["arms"]["reprompt"]["over_rerun"] == pytest.approx(0.0, abs=1e-9)
        assert "does not beat a blind re-run" in render(data)

    def test_without_a_baseline_the_report_says_the_gain_is_unattributable(self):
        data = compare({"reprompt": arm("reprompt", [(0.5, 0.9)])})
        assert data["baseline"] is None
        assert "no blind re-run arm was run" in render(data)


class TestFailuresAndSlices:
    def test_an_arm_that_errored_keeps_the_original_answer(self):
        """Repair is an improvement, not a dependency. A crashed attempt must leave the
        document where it was, never at zero -- scoring a failed call as a ruined
        extraction would make an outage look like a damaging loop."""
        score = RepairScore()
        score.add(outcome(0.8, 0.8, error="timeout", attempts=1))
        data = score.to_dict()
        assert data["errors"] == 1
        assert data["damaged"] == 0
        assert data["net_delta"] == pytest.approx(0.0, abs=1e-9)

    def test_slices_show_which_half_an_average_is_hiding(self):
        score = RepairScore()
        score.add(outcome(0.5, 0.9, doc_type="invoices"))
        score.add(outcome(0.9, 0.5, doc_type="resumes"))
        rows = {r["doc_type"]: r for r in by_slice(score, "doc_type")}
        assert rows["invoices"]["improved"] == 1
        assert rows["resumes"]["damaged"] == 1
        assert score.to_dict()["net_delta"] == pytest.approx(0.0, abs=1e-9)

    def test_an_empty_arm_renders_rather_than_raising(self):
        assert "no documents" in render(compare({"reprompt": RepairScore("reprompt")}))
