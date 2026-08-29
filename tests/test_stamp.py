"""Result provenance, and the questions it answers without archaeology.

Twice in Phase 6 a bug invalidated a set of numbers, and both times the expensive part
was working out which artifacts on disk had inherited it -- done by reading commit
timestamps against file mtimes. These tests pin the facts that make it a query instead.

The organising idea is that four things move independently -- the code, the meaning of
the metric, the corpus (in two senses), and the cohort -- and a hash of one says nothing
about the others.
"""
import json

import pytest

from core import stamp as stamp_mod


def stamped(version="phase6-v2", label="L", docs="D", inputs="I", count=75):
    return {"evaluation_version": version,
            "corpus": {"label": label, "document_set": docs, "input": inputs,
                       "document_count": count}}


class TestItFailsClosed:
    """A silent denominator change is dangerous precisely because the metric that
    results is still perfectly plausible."""

    def test_a_different_cohort_is_refused(self):
        ok, why, _ = stamp_mod.comparable(stamped(docs="aaa", count=75),
                                          stamped(docs="bbb", count=74))
        assert not ok
        assert "different cohorts" in why

    def test_different_labels_are_refused(self):
        ok, why, _ = stamp_mod.comparable(stamped(label="one"), stamped(label="two"))
        assert not ok and "different corpus labels" in why

    def test_a_different_evaluation_version_is_refused(self):
        ok, why, _ = stamp_mod.comparable(stamped(version="phase6-v1"),
                                          stamped(version="phase6-v2"))
        assert not ok and "changed meaning" in why

    def test_unstamped_is_refused_rather_than_assumed_current(self):
        ok, why, _ = stamp_mod.comparable({}, stamped())
        assert not ok and "unstamped" in why

    def test_a_commit_difference_alone_does_not_refuse(self):
        """Most commits are comments and renderer tweaks. A flag that fires on every
        commit gets ignored on the one that mattered."""
        one, two = stamped(), stamped()
        one["code"] = {"commit": "111"}
        two["code"] = {"commit": "222"}
        assert stamp_mod.comparable(one, two)[0]


class TestInputsAndLabelsAreDifferentQuestions:
    """An earlier version hashed labels only, reasoning that PDF bytes churn on every
    rebuild. That avoided a false alarm by discarding a true one: a corpus regenerated
    from the same seed can produce different pixels with identical labels, and
    extraction can legitimately move because of it."""

    def test_same_labels_different_inputs_is_a_note_not_a_refusal(self):
        ok, why, note = stamp_mod.comparable(stamped(inputs="one"),
                                             stamped(inputs="two"))
        assert ok and not why
        assert "regenerated" in note and "different pixels" in note

    def test_identical_everything_produces_no_note(self):
        ok, why, note = stamp_mod.comparable(stamped(), stamped())
        assert ok and not why and not note

    def test_the_cohort_hash_is_order_independent(self, tmp_path):
        for name in ("b.pdf", "a.pdf"):
            (tmp_path / name).write_bytes(b"x")
        one = stamp_mod.cohort_fingerprint(str(tmp_path), ["a.pdf", "b.pdf"])
        two = stamp_mod.cohort_fingerprint(str(tmp_path), ["b.pdf", "a.pdf"])
        assert one["document_set"] == two["document_set"]
        assert one["input"] == two["input"]

    def test_changing_a_document_changes_the_input_hash_only(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"one")
        first = stamp_mod.cohort_fingerprint(str(tmp_path), ["a.pdf"])
        (tmp_path / "a.pdf").write_bytes(b"two")
        second = stamp_mod.cohort_fingerprint(str(tmp_path), ["a.pdf"])
        assert first["document_set"] == second["document_set"]
        assert first["input"] != second["input"]

    def test_a_cohort_that_cannot_be_read_in_full_says_so(self, tmp_path):
        """The count would otherwise be a promise the input hash quietly breaks."""
        (tmp_path / "a.pdf").write_bytes(b"one")
        out = stamp_mod.cohort_fingerprint(str(tmp_path), ["a.pdf", "gone.pdf"])
        assert out["document_count"] == 2
        assert out["documents_missing"] == 1
        assert "DOCUMENTS MISSING" in stamp_mod.describe({"corpus": out})


class TestTheLabelFingerprint:
    def test_it_ignores_the_documents(self, tmp_path):
        """Hashing PDFs into the LABEL fingerprint would report an evaluation-target
        change on every rebuild. The document bytes are tracked separately, by the
        cohort hash, where they mean something different."""
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "invoices.json").write_text('[{"file": "a.pdf"}]', encoding="utf-8")
        (tmp_path / "a.pdf").write_bytes(b"one")
        first = stamp_mod.label_fingerprint(str(tmp_path))
        (tmp_path / "a.pdf").write_bytes(b"entirely different bytes")
        assert stamp_mod.label_fingerprint(str(tmp_path)) == first

    def test_changing_an_answer_changes_it(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "invoices.json").write_text('[{"total": "1.00"}]', encoding="utf-8")
        first = stamp_mod.label_fingerprint(str(tmp_path))
        (labels / "invoices.json").write_text('[{"total": "2.00"}]', encoding="utf-8")
        assert stamp_mod.label_fingerprint(str(tmp_path)) != first

    def test_no_labels_is_none_not_a_crash(self, tmp_path):
        assert stamp_mod.label_fingerprint(str(tmp_path)) is None


class TestKnownGaps:
    def test_the_seed_is_recorded_as_absent_rather_than_omitted(self):
        """The generator takes --seed and writes nothing about it into the corpus, so a
        rebuilt set cannot say which seed made it. An omitted field reads as "not
        applicable"; this one is "not captured"."""
        out = stamp_mod.generator_fingerprint()
        assert "seed" in out and out["seed"] is None

    def test_the_generator_sources_are_hashed(self):
        out = stamp_mod.generator_fingerprint()
        assert out.get("generator_sources")


class TestItStampsRealReports:
    def test_the_stamp_is_json_serialisable(self):
        json.dumps(stamp_mod.stamp("data/degraded"))

    def test_the_version_matches_the_history_note(self):
        assert stamp_mod.EVALUATION == "phase6-v2"
        with open(stamp_mod.__file__, encoding="utf-8") as handle:
            assert "phase6-v2" in handle.read()

    def test_dirty_is_recorded_not_refused(self):
        out = stamp_mod.stamp()
        assert isinstance(out["code"]["dirty"], bool)

    def test_extra_fields_travel(self):
        out = stamp_mod.stamp("", {"arms": ["rerun"], "budget": 3})
        assert out["arms"] == ["rerun"] and out["budget"] == 3

    def test_describe_refuses_to_dress_up_an_unstamped_result(self):
        assert "incomparable" in stamp_mod.describe({})
