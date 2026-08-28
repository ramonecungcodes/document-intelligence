"""Result provenance, and the two questions it exists to answer without archaeology.

Twice in Phase 6 a bug invalidated a set of numbers, and both times the expensive part
was working out which artifacts on disk had inherited it -- done by reading commit
timestamps against file mtimes, which is archaeology. These tests pin the two facts that
make it a query instead.
"""
import json

import pytest

from core import stamp as stamp_mod


class TestTheVersionIsTheDecidingField:
    def test_two_evaluation_versions_are_never_comparable(self):
        """A version change is a statement that a number changed meaning. That is
        exactly the case this exists for: repair results before phase6-v2 overstate
        damage, and no commit hash says so on its own."""
        ok, why = stamp_mod.comparable(
            {"evaluation_version": "phase6-v1", "labels": "abc"},
            {"evaluation_version": "phase6-v2", "labels": "abc"})
        assert not ok
        assert "changed meaning" in why

    def test_a_commit_difference_alone_does_not_block_comparison(self):
        """Most commits are comments and renderer tweaks. Treating every one as
        invalidating would make the flag meaningless and it would be ignored."""
        ok, _ = stamp_mod.comparable(
            {"evaluation_version": "phase6-v2", "labels": "abc", "commit": "111"},
            {"evaluation_version": "phase6-v2", "labels": "abc", "commit": "222"})
        assert ok

    def test_different_labels_block_comparison(self):
        """report.json overstated field accuracy by three and a half points for weeks
        because the corpus was rebuilt underneath it. The code was correct; the labels
        had changed, and a commit hash cannot see that."""
        ok, why = stamp_mod.comparable(
            {"evaluation_version": "phase6-v2", "labels": "abc"},
            {"evaluation_version": "phase6-v2", "labels": "def"})
        assert not ok
        assert "different corpus labels" in why

    def test_an_unstamped_result_is_incomparable_not_assumed_current(self):
        ok, why = stamp_mod.comparable({}, {"evaluation_version": "phase6-v2"})
        assert not ok
        assert "unstamped" in why


class TestTheCorpusFingerprint:
    def test_it_hashes_labels_and_not_documents(self, tmp_path):
        """A corpus regenerated from the same seed can differ byte-for-byte in its PDFs
        without a single answer changing. Hashing those would report a change on every
        rebuild and the signal would be discarded as noise."""
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "invoices.json").write_text('[{"file": "a.pdf"}]', encoding="utf-8")
        (tmp_path / "invoices").mkdir()
        (tmp_path / "invoices" / "a.pdf").write_bytes(b"one")
        first = stamp_mod.corpus_fingerprint(str(tmp_path))
        (tmp_path / "invoices" / "a.pdf").write_bytes(b"two entirely different")
        assert stamp_mod.corpus_fingerprint(str(tmp_path))["labels"] == first["labels"]

    def test_changing_a_label_changes_the_fingerprint(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        (labels / "invoices.json").write_text('[{"total": "1.00"}]', encoding="utf-8")
        first = stamp_mod.corpus_fingerprint(str(tmp_path))
        (labels / "invoices.json").write_text('[{"total": "2.00"}]', encoding="utf-8")
        assert stamp_mod.corpus_fingerprint(str(tmp_path))["labels"] != first["labels"]

    def test_a_corpus_with_no_labels_is_reported_not_crashed(self, tmp_path):
        out = stamp_mod.corpus_fingerprint(str(tmp_path))
        assert out["labels"] is None


class TestDirtyIsRecordedNotRefused:
    def test_the_stamp_carries_a_dirty_flag(self):
        """Refusing to stamp a dirty tree would block the exploratory runs that are
        most of the work. But a commit that does not identify the code is worth
        knowing, so it travels with the result."""
        out = stamp_mod.stamp()
        assert "dirty" in out
        assert isinstance(out["dirty"], bool)

    def test_describe_says_so_loudly(self):
        line = stamp_mod.describe({"evaluation_version": "x", "commit": "abc",
                                   "dirty": True})
        assert "DIRTY" in line

    def test_describe_refuses_to_dress_up_an_unstamped_result(self):
        assert "incomparable" in stamp_mod.describe({})


class TestItActuallyStampsRealReports:
    def test_the_stamp_is_json_serialisable(self):
        """It is written into every report, so anything unserialisable would break the
        write rather than the stamp."""
        json.dumps(stamp_mod.stamp("data/degraded"))

    def test_the_current_version_is_the_post_fix_one(self):
        """If this is ever bumped without the history note being updated, the note is
        what a future reader relies on to know what changed."""
        assert stamp_mod.EVALUATION == "phase6-v2"
        assert "phase6-v2" in stamp_mod.__doc__ or "phase6-v2" in open(
            stamp_mod.__file__, encoding="utf-8").read()

    def test_extra_fields_travel(self):
        out = stamp_mod.stamp("", {"arms": ["rerun"], "budget": 3})
        assert out["arms"] == ["rerun"] and out["budget"] == 3
