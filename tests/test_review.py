"""The review record, and the properties that make it trainable later.

Every test here guards a decision that cannot be made retroactively. A review dataset
collected without these is not a worse dataset, it is an unusable one -- and it will not
look unusable, because a model fitted on it scores well on the only documents it was
ever shown.
"""
import json

import pytest

from route import review as review_mod
from route.review import (BY_EXPLORATION, BY_GATE, CONFIRMED, CORRECTED, FieldReview,
                          INVENTED, Review, UNSURE, explore, training_rows)


class TestExplorationIsStableAndReal:
    def test_the_same_document_is_always_sampled_or_never(self):
        """Hashed, not drawn. A fresh draw per run would audit a document twice and
        never, and the exploration rate could not be reasoned about across runs."""
        first = [explore(f"invoices/{i}.pdf", 0.3) for i in range(200)]
        second = [explore(f"invoices/{i}.pdf", 0.3) for i in range(200)]
        assert first == second

    def test_the_rate_is_approximately_honoured(self):
        sampled = sum(explore(f"invoices/{i}.pdf", 0.2) for i in range(4000))
        assert 0.17 < sampled / 4000 < 0.23

    def test_a_different_seed_draws_a_different_sample(self):
        a = {f for f in (f"invoices/{i}.pdf" for i in range(400))
             if explore(f, 0.25, seed="one")}
        b = {f for f in (f"invoices/{i}.pdf" for i in range(400))
             if explore(f, 0.25, seed="two")}
        assert a != b

    def test_zero_means_off_and_one_means_everything(self):
        assert not explore("invoices/a.pdf", 0.0)
        assert explore("invoices/a.pdf", 1.0)


class TestTheActionIsNotTheTruth:
    def test_correct_is_derived_from_the_action(self):
        """Stored separately they could disagree, and a row that says both
        `confirmed` and `incorrect` would train a scorer on a contradiction."""
        assert FieldReview("total", CONFIRMED).correct
        assert not FieldReview("total", CORRECTED, INVENTED).correct

    def test_unsure_is_unlabelled_rather_than_incorrect(self):
        """A reviewer who could not tell has not said the extractor was wrong. Folding
        that into `incorrect` would teach a scorer that hard documents are wrong
        documents."""
        row = FieldReview("total", UNSURE, "ambiguous")
        assert not row.correct
        assert not row.labelled

    def test_a_confirmed_field_cannot_carry_a_failure_reason(self):
        with pytest.raises(ValueError, match="confirmed"):
            FieldReview("total", CONFIRMED, INVENTED)

    def test_a_changed_field_must_say_why(self):
        """Binary correct/incorrect makes an invented value and a missed one the same
        number, and they are opposite errors with opposite costs."""
        with pytest.raises(ValueError, match="needs a reason"):
            FieldReview("total", CORRECTED)

    def test_unknown_actions_and_reasons_are_refused(self):
        with pytest.raises(ValueError, match="unknown action"):
            FieldReview("total", "looked_fine")
        with pytest.raises(ValueError, match="unknown reason"):
            FieldReview("total", CORRECTED, "vibes")


class TestRubberStampingStaysMeasurable:
    def test_review_duration_is_kept(self):
        """A queue reviewed at two seconds a document produced attendance, not labels.
        Without this recorded there is no way to find that out afterwards."""
        row = Review("invoices/a.pdf", seconds=1.5,
                     fields=[FieldReview("total", CONFIRMED)]).to_dict()
        assert row["seconds"] == 1.5

    def test_the_reviewer_is_kept(self):
        row = Review("invoices/a.pdf", reviewer="rc",
                     fields=[FieldReview("total", CONFIRMED)]).to_dict()
        assert row["reviewer"] == "rc"


class TestSelectionSurvivesToTraining:
    def queue_and_outcomes(self, tmp_path, selection):
        queue = tmp_path / "q.jsonl"
        queue.write_text(json.dumps({
            "file": "invoices/a.pdf", "doc_type": "invoices", "profile": "clean",
            "selection": selection, "why": "blank_share 0.5 above 0.2",
            "signals": {"blank_share": 0.5, "ocr_confidence": None},
        }) + "\n", encoding="utf-8")
        review_mod.write(str(queue), [Review(
            "invoices/a.pdf", selection=selection, seconds=40,
            fields=[FieldReview("total", CONFIRMED),
                    FieldReview("vendor_name", CORRECTED, INVENTED,
                                was="ACME", now=None),
                    FieldReview("po_number", UNSURE, "ambiguous")])])
        return str(queue)

    def test_every_row_carries_how_the_document_was_selected(self, tmp_path):
        """The field that decides whether the dataset can be corrected for bias. A
        dataset that has forgotten which rows came from exploration cannot be
        de-biased, and de-biasing is the entire reason exploration costs money."""
        rows = training_rows(self.queue_and_outcomes(tmp_path, BY_EXPLORATION))
        assert rows and all(r["selection"] == BY_EXPLORATION for r in rows)

    def test_unsure_fields_are_dropped_not_guessed(self, tmp_path):
        rows = training_rows(self.queue_and_outcomes(tmp_path, BY_GATE))
        assert {r["field"] for r in rows} == {"total", "vendor_name"}
        assert [r["correct"] for r in rows if r["field"] == "total"] == [1]
        assert [r["correct"] for r in rows if r["field"] == "vendor_name"] == [0]

    def test_the_failure_reason_reaches_the_training_row(self, tmp_path):
        rows = training_rows(self.queue_and_outcomes(tmp_path, BY_GATE))
        reasons = {r["field"]: r["reason"] for r in rows}
        assert reasons["vendor_name"] == INVENTED

    def test_signals_come_from_the_queue_not_from_recomputing(self, tmp_path):
        """These are the values as they stood when the decision was made. Recomputing
        them at training time would use whatever the feature code does now, which is
        not what the router saw and not what the label describes."""
        rows = training_rows(self.queue_and_outcomes(tmp_path, BY_GATE))
        assert rows[0]["signals"]["blank_share"] == 0.5
        assert rows[0]["signals"]["ocr_confidence"] is None

    def test_one_row_per_field_not_per_document(self, tmp_path):
        rows = training_rows(self.queue_and_outcomes(tmp_path, BY_GATE))
        assert len(rows) == 2
        assert len({r["field"] for r in rows}) == 2


class TestTheFileContract:
    def test_outcomes_live_beside_the_queue_not_inside_it(self, tmp_path):
        queue = tmp_path / "review-queue.jsonl"
        assert review_mod.path_for(str(queue)).endswith("review-queue.outcomes.jsonl")

    def test_no_outcomes_yet_is_not_an_error(self, tmp_path):
        queue = tmp_path / "q.jsonl"
        queue.write_text("", encoding="utf-8")
        assert review_mod.read(str(queue)) == {}
        assert training_rows(str(queue)) == []

    def test_a_stale_format_raises_rather_than_being_read_as_current(self, tmp_path):
        queue = tmp_path / "q.jsonl"
        queue.write_text("", encoding="utf-8")
        out = tmp_path / "q.outcomes.jsonl"
        out.write_text(json.dumps({"format": 99, "file": "a.pdf", "fields": []}) + "\n",
                       encoding="utf-8")
        with pytest.raises(ValueError, match="format 99"):
            review_mod.read(str(queue))

    def test_an_unknown_selection_is_refused(self):
        with pytest.raises(ValueError, match="unknown selection"):
            Review("a.pdf", selection="because")
