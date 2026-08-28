"""The routing policy, and the ways a gate could quietly do the wrong thing.

A routing policy fails in two directions and only one of them is visible in production.
Accepting a bad document shows up later as a wrong record; sending a perfect document to
a person shows up as nothing at all, because the person fixes nothing and moves on. So
most of these tests are about the second direction.
"""
import pytest

from route.policy import ACCEPT, GATES, REVIEW, Policy, build


def policy(**thresholds):
    """A policy with every gate off except the ones named."""
    off = {"classifier_floor": 0.0, "blank_share_ceiling": 0.0,
           "ocr_confidence_floor": 0.0, "words_floor": 0.0,
           "validator_errors_ceiling": -1}
    off.update(thresholds)
    return Policy(**off)


class TestMissingIsNotBad:
    def test_a_signal_nobody_recorded_does_not_fire_a_gate(self):
        """The failure this exists to prevent: a clean page has no OCR confidence, and
        reading absence as zero would route the entire clean corpus to a person on the
        grounds that it was illegible."""
        decision = policy(ocr_confidence_floor=0.8).decide({"ocr_confidence": None})
        assert decision.action == ACCEPT

    def test_an_empty_signal_set_accepts(self):
        assert policy(blank_share_ceiling=0.2).decide({}).action == ACCEPT
        assert policy(blank_share_ceiling=0.2).decide(None).action == ACCEPT

    def test_a_real_zero_still_fires(self):
        """Absent and zero must not be conflated in either direction. A page that
        genuinely yielded no words is exactly what the gate is for."""
        assert policy(words_floor=10).decide({"words_per_page": 0.0}).action == REVIEW


class TestGatesAreIndependent:
    def test_any_gate_routes_the_document(self):
        chosen = policy(blank_share_ceiling=0.2, validator_errors_ceiling=0)
        assert chosen.decide({"blank_share": 0.5, "validator_errors": 0.0}).review
        assert chosen.decide({"blank_share": 0.0, "validator_errors": 2.0}).review
        assert not chosen.decide({"blank_share": 0.1, "validator_errors": 0.0}).review

    def test_every_reason_is_recorded_not_just_the_first(self):
        """A reviewer needs all of them. Reporting only the first gate to fire would
        make a document look like a marginal case when three things were wrong."""
        decision = policy(blank_share_ceiling=0.2,
                          validator_errors_ceiling=0).decide(
            {"blank_share": 0.9, "validator_errors": 3.0})
        assert len(decision.reasons) == 2
        assert {r.gate for r in decision.reasons} == {"blank_share", "validator_errors"}

    def test_the_reason_says_what_a_person_can_check(self):
        decision = policy(blank_share_ceiling=0.2).decide({"blank_share": 0.42})
        assert str(decision.reasons[0]) == "blank_share 0.42 above 0.2"

    def test_every_declared_gate_is_reachable(self):
        """A gate whose setting name is misspelled would silently never fire, and the
        policy would look stricter than it is."""
        names = {setting for _signal, _direction, setting in GATES}
        assert names <= set(Policy().thresholds)
        for signal, direction, setting in GATES:
            value = 999.0 if direction == "above" else -999.0
            chosen = policy(**{setting: 0.5 if setting !=
                               "validator_errors_ceiling" else 0})
            assert chosen.decide({signal: value}).review, f"{setting} never fires"


class TestSwitchingGatesOff:
    def test_zero_disables_a_floor_or_ceiling(self):
        """"Confidence under zero" and "fewer than zero words" describe no document,
        so zero is the natural off switch for those."""
        assert not policy(classifier_floor=0.0).decide(
            {"classifier_confidence": 0.01}).review
        assert not policy(blank_share_ceiling=0.0).decide({"blank_share": 1.0}).review

    def test_zero_errors_means_the_validator_gate_is_on(self):
        """The asymmetry, pinned. "More than zero errors" describes exactly the
        documents a validator exists to find, so zero has to mean on -- and if this
        ever flipped to matching the others, every failing document would be accepted
        and nothing would look broken."""
        assert policy(validator_errors_ceiling=0).decide(
            {"validator_errors": 1.0}).review
        assert not policy(validator_errors_ceiling=0).decide(
            {"validator_errors": 0.0}).review
        assert not policy(validator_errors_ceiling=-1).decide(
            {"validator_errors": 99.0}).review

    def test_a_policy_with_nothing_on_says_so(self):
        assert "every document accepted" in policy().describe()


class TestTheManifestIsTheSourceOfTruth:
    def test_thresholds_come_from_the_manifest(self):
        from core import config as config_mod

        chosen = build(config_mod.load("di.toml"))
        # The measured design-holdout floor. If this drifts from the number in
        # [classifiers.cascade] the two stages disagree about the same decision.
        assert chosen.thresholds["classifier_floor"] == pytest.approx(0.85)
        assert chosen.thresholds["blank_share_ceiling"] > 0

    def test_the_router_is_a_pipeline_slot(self):
        """Routing is a stage, not a script beside the pipeline. If `router` were not
        in SLOTS the manifest block would parse to nothing and every threshold would
        silently fall back to a default."""
        from core.config import SLOTS

        assert "router" in SLOTS
        assert SLOTS.index("router") > SLOTS.index("validator")

    def test_overrides_beat_the_manifest(self):
        from core import config as config_mod

        chosen = build(config_mod.load("di.toml"), {"classifier_floor": 0.5})
        assert chosen.thresholds["classifier_floor"] == pytest.approx(0.5)
