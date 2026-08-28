"""The repair plugins, and what their prompts must never say.

A repair prompt is the one place in this pipeline where a stage argues with the model
about an answer it already gave. Most of the care went into what the prompt refuses to
claim, so most of these tests assert absences.
"""
import json

import pytest

import repair
from core.doctypes import REGISTRY
from repair.base import Complaint, Context, Repaired, Repairer, complaints_for


class Backend:
    """A backend that records what it was asked and replies with whatever it is given."""

    def __init__(self, reply=None, error=""):
        self.reply = reply if reply is not None else {"total": "1.00"}
        self.error = error
        self.calls = []

    def complete(self, system, user, schema=None, **_):
        self.calls.append({"system": system, "user": user, "schema": schema})
        return type("C", (), {"error": self.error,
                              "text": json.dumps(self.reply),
                              "truncated": False, "usage": None})()


def context(backend, record=None, complaints=(), doctype=None):
    spec = doctype or REGISTRY["invoice"]
    return Context(backend=backend, doctype=spec, variant="",
                   path="data/invoices/a.pdf", relative_path="invoices/a.pdf",
                   record=record or {"file": "invoices/a.pdf", "total": "9.99"},
                   text="INVOICE\nTotal 9.99", complaints=list(complaints))


class TestTheBaselineIsTrulyBlind:
    def test_rerun_never_sees_the_complaints(self):
        """The whole comparison rests on this. If the baseline learned anything from
        the complaints it would stop being a baseline, and the difference between the
        arms would price nothing."""
        backend = Backend()
        arm = repair.build("rerun")
        arm.repair(context(backend, complaints=[
            Complaint("validator", "totals_mismatch", "subtotal + tax != total")]))
        sent = backend.calls[0]["user"]
        assert "totals_mismatch" not in sent
        assert "subtotal" not in sent.replace("INVOICE\nTotal 9.99", "")

    def test_rerun_never_sees_the_previous_answer(self):
        backend = Backend()
        repair.build("rerun").repair(
            context(backend, record={"file": "a.pdf", "total": "SENTINEL"}))
        assert "SENTINEL" not in backend.calls[0]["user"]

    def test_rerun_sends_the_same_request_as_the_extractor(self):
        """A baseline that drifts from the thing it baselines is a second undocumented
        extractor. Both must come from the same schema helpers."""
        from extract import schema as schema_mod

        backend = Backend()
        repair.build("rerun").repair(context(backend))
        call = backend.calls[0]
        assert call["system"] == schema_mod.instructions(REGISTRY["invoice"], "")
        assert call["schema"] == schema_mod.json_schema(REGISTRY["invoice"], "")


class TestWhatTheGuidedPromptRefusesToSay:
    def build(self, complaints, **kw):
        backend = Backend()
        repair.build("reprompt", overrides=kw).repair(
            context(backend, complaints=complaints))
        return backend.calls[0]["user"]

    def test_it_does_not_tell_the_model_which_field_is_wrong(self):
        """A validator knows subtotal + tax != total. It does not know which of the
        three was misread, and neither do we -- naming one is an instruction to change
        a field that may have been the only correct one."""
        sent = self.build([Complaint("validator", "totals_mismatch",
                                     "subtotal + tax != total",
                                     ("subtotal", "tax", "total"))])
        assert "subtotal + tax != total" in sent
        assert "do not say which value is wrong" not in sent.lower()
        assert "They do not say which field caused it" in sent

    def test_it_forbids_blanking_a_field_to_satisfy_a_check(self):
        """The shortest path to a silent validator is an empty field. The scorer
        catches this after the fact; the prompt is the cheaper place to prevent it."""
        sent = self.build([Complaint("validator", "totals_mismatch", "x")])
        assert "Do not remove a value to satisfy a check" in sent

    def test_it_forbids_inventing_a_field(self):
        """A blank gets looked at; a confident wrong value flows downstream. Pushing
        the model to fill gaps trades the cheap error for the expensive one."""
        sent = self.build([Complaint("router", "blank_share", "blank_share was 0.75")])
        assert "Do not invent a value" in sent

    def test_it_allows_the_model_to_stand_its_ground(self):
        sent = self.build([Complaint("validator", "x", "y")])
        assert "return it unchanged" in sent

    def test_harness_provenance_is_stripped_from_the_previous_answer(self):
        """`_error` and `_normalizer` are things this pipeline attached, not things the
        model said. Quoting them back invites reasoning about the harness, and `_error`
        in particular would read as a claim about the page."""
        backend = Backend()
        repair.build("reprompt").repair(context(
            backend,
            record={"file": "invoices/a.pdf", "total": "9.99",
                    "_error": "truncated", "_normalizer": {"engine": "native"}},
            complaints=[Complaint("validator", "x", "y")]))
        sent = backend.calls[0]["user"]
        assert "9.99" in sent
        assert "_error" not in sent and "truncated" not in sent
        assert "_normalizer" not in sent

    def test_with_no_complaints_it_declines_rather_than_inventing_one(self):
        """Saying "checks flagged this" with no checks is a falsehood the model has to
        resolve somehow, and it resolves it by changing something."""
        backend = Backend()
        result = repair.build("reprompt").repair(context(backend, complaints=[]))
        assert backend.calls == []
        assert not result.changed

    def test_complaints_are_capped_and_the_remainder_is_counted(self):
        many = [Complaint("validator", f"code_{i}", f"message {i}") for i in range(20)]
        sent = self.build(many, max_complaints=3)
        assert "message 0" in sent and "message 2" in sent
        assert "message 9" not in sent
        assert "and 17 more" in sent


class TestBounds:
    def test_the_budget_is_respected(self):
        class Failing(Repairer):
            def __init__(self, max_attempts):
                self.max_attempts = max_attempts
                self.tries = 0

            def attempt(self, ctx):
                self.tries += 1
                raise RuntimeError("nope")

        arm = Failing(max_attempts=3)
        result = arm.repair(context(Backend()))
        assert arm.tries == 3
        assert not result.changed
        assert "nope" in result.error

    def test_it_stops_at_the_first_usable_record(self):
        """Best-of-N would need a judge, the only judge available is the validators,
        and picking the attempt that satisfies them is exactly the optimisation the
        scorer was written to catch."""
        class Counting(Repairer):
            def __init__(self):
                self.max_attempts = 5
                self.tries = 0

            def attempt(self, ctx):
                self.tries += 1
                return {"total": "1.00"}

        arm = Counting()
        assert arm.repair(context(Backend())).attempts == 1
        assert arm.tries == 1

    def test_a_backend_error_leaves_the_original_answer_standing(self):
        """Repair is an improvement, not a dependency."""
        result = repair.build("rerun").repair(context(Backend(error="503")))
        assert not result.changed
        assert result.record is None
        assert "503" in result.error


class TestComplaintsAreGathered:
    def test_warnings_do_not_become_complaints(self):
        """Measured at lift +0.001 against extraction quality in Phase 5: warnings
        carry nothing, and prompting a model to act on one asks it to change an answer
        for no reason."""
        class Finding:
            def __init__(self, severity):
                self.code = "c"
                self.severity = severity
                self.message = "m"
                self.fields = ()

        class Report:
            findings = [Finding("warning"), Finding("error")]

        import validate.base as vb
        original = vb.run
        vb.run = lambda *a, **k: Report()
        try:
            out = complaints_for({"file": "a"}, REGISTRY["invoice"], "",
                                 validators=[object()])
        finally:
            vb.run = original
        assert len(out) == 1

    def test_router_reasons_arrive_as_complaints_about_suspicion(self):
        """A gate is a statement about the answer, not about the page, and must not be
        phrased as though the document said something."""
        from route.policy import Policy

        decision = Policy(blank_share_ceiling=0.2).decide({"blank_share": 0.75})
        out = complaints_for({"file": "a"}, None, "", None, decision)
        assert len(out) == 1
        assert out[0].source == "router"
        assert "0.75" in str(out[0]) and "threshold" in str(out[0])


class TestRegistration:
    def test_both_arms_are_registered_and_build_from_the_manifest(self):
        from core import config as config_mod

        assert {"rerun", "reprompt"} <= set(repair.REPAIRERS)
        chosen = config_mod.load("di.toml")
        for name in ("rerun", "reprompt"):
            assert repair.build(name, chosen).max_attempts >= 1

    def test_an_unknown_repairer_names_the_ones_that_exist(self):
        with pytest.raises(SystemExit) as caught:
            repair.build("nonesuch")
        assert "rerun" in str(caught.value)

    def test_repairer_is_a_pipeline_slot(self):
        from core.config import SLOTS

        assert "repairer" in SLOTS
        assert SLOTS.index("repairer") > SLOTS.index("validator")
