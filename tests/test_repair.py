"""Repair scoring, and the two numbers that must never be quotable alone.

A repair loop is the first stage in this project that can improve its own score by
making documents worse, so its scorer is written adversarially: every test here is a way
a loop could look successful without helping.
"""
import pytest

from eval.repair import (Outcome, RepairScore, by_slice, compare, paired,
                         render)


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
        pairs = [(0.5, 0.7)] * 20
        data = compare({"rerun": arm("rerun", pairs),
                        "reprompt": arm("reprompt", pairs)})
        assert data["arms"]["reprompt"]["over_rerun"] == pytest.approx(0.0, abs=1e-9)
        assert not data["paired"]["reprompt"]["resolvable"]
        assert "not measured" in render(data)

    def test_an_effect_inside_its_own_error_bar_is_not_quotable(self):
        """The finding that made this necessary: two runs of the same comparison came
        out with opposite signs. That is not a contradiction, it is what an effect
        smaller than its own interval looks like when you run it twice. The report has
        to refuse the point estimate rather than print it with a caveat."""
        rerun = arm("rerun", [(0.5, 0.5 + d) for d in
                              (0.4, -0.3, 0.2, -0.1, 0.3, -0.4, 0.1, -0.2)])
        guided = arm("reprompt", [(0.5, 0.5 + d) for d in
                                  (-0.3, 0.4, -0.1, 0.3, -0.4, 0.2, -0.2, 0.15)])
        pair = compare({"rerun": rerun, "reprompt": guided})["paired"]["reprompt"]
        assert not pair["resolvable"]
        assert pair["interval"][0] < 0 < pair["interval"][1]

    def test_a_real_difference_is_declared_resolvable(self):
        """The other direction must also work, or the report would call everything
        unmeasured and be useless."""
        rerun = arm("rerun", [(0.5, 0.5)] * 30)
        guided = arm("reprompt", [(0.5, 0.8)] * 30)
        pair = compare({"rerun": rerun, "reprompt": guided})["paired"]["reprompt"]
        assert pair["resolvable"]
        assert pair["mean"] == pytest.approx(0.3, abs=1e-4)
        assert "worth something beyond a second sample" in render(
            compare({"rerun": rerun, "reprompt": guided}))

    def test_without_a_baseline_the_report_says_the_gain_is_unattributable(self):
        data = compare({"reprompt": arm("reprompt", [(0.5, 0.9)])})
        assert data["baseline"] is None
        assert "no blind re-run arm" in render(data)


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


class TestAWinBetweenTwoLosersIsNotAWin:
    """The degraded-corpus result, reduced to its arithmetic.

    Both arms left documents worse than they were found, and the guided arm was
    resolvably better than the blind one. Every individual statement in that sentence
    is true, and "reprompt beats the baseline, p<0.05" is the one that gets quoted.
    """

    def losers(self):
        # Both arms harmful; the guided one consistently less so.
        rerun = arm("rerun", [(0.6, 0.4)] * 20)
        guided = arm("reprompt", [(0.6, 0.5)] * 20)
        return compare({"rerun": rerun, "reprompt": guided})

    def test_the_paired_win_is_still_reported(self):
        data = self.losers()
        pair = data["paired"]["reprompt"]
        assert pair["resolvable"] and pair["mean"] > 0

    def test_but_it_is_qualified_in_the_same_breath(self):
        text = render(self.losers())
        assert "less harmful" in text
        assert "Neither should run on these documents." in text

    def test_a_net_negative_arm_is_called_out_above_the_fold(self):
        """Not in a footnote. An arm scoring below zero should be off, and that has to
        be the first thing read after the table."""
        text = render(self.losers())
        assert "NET-NEGATIVE" in text
        headline = text.index("NET-NEGATIVE")
        assert headline < text.index("SECONDARY")

    def test_two_healthy_arms_are_not_warned_about(self):
        """The warning must not fire on a genuinely good result, or it becomes noise
        that is scrolled past when it matters."""
        data = compare({"rerun": arm("rerun", [(0.5, 0.6)] * 20),
                        "reprompt": arm("reprompt", [(0.5, 0.8)] * 20)})
        text = render(data)
        assert "NET-NEGATIVE" not in text
        assert "less harmful" not in text

    def test_gates_are_kept_per_document(self):
        """So "the gates went quiet while the documents got worse" is checkable one
        document at a time rather than inferred from two totals."""
        score = RepairScore()
        score.add(outcome(0.9, 0.3, gates_before=2, gates_after=0,
                          file="invoices/x.pdf"))
        data = score.to_dict()
        assert data["gates"]["invoices/x.pdf"] == [2, 0]
        assert data["deltas"]["invoices/x.pdf"] < 0


class TestTheGoodhartCrossTab:
    """Whether the gates went quiet on the same documents that got worse.

    The aggregate -- 52 cleared, 52 damaged -- is suggestive and proves nothing. Two
    disjoint groups of 52 give identical totals and mean something entirely different.
    """

    def test_disjoint_groups_are_not_reported_as_silencing(self):
        """The claim this guards. Same totals, innocent structure: the documents that
        cleared their gates are not the documents that got worse."""
        from eval.repair import goodhart

        score = RepairScore()
        for i in range(5):        # cleared, and improved
            score.add(outcome(0.4, 0.9, 2, 0, file=f"a{i}.pdf"))
        for i in range(5):        # damaged, gates still firing
            score.add(outcome(0.9, 0.4, 2, 2, file=f"b{i}.pdf"))
        table = goodhart(score.to_dict())
        assert table["cleared_and_damaged"] == 0
        assert table["cleared_and_improved"] == 5
        assert table["held_and_damaged"] == 5
        assert table["lift"] in (None, 0.0) or table["lift"] < 1.0

    def test_the_damning_cell_is_counted_and_named(self):
        from eval.repair import goodhart

        score = RepairScore()
        for i in range(6):        # cleared every rule AND got worse
            score.add(outcome(0.9, 0.3, 2, 0, file=f"a{i}.pdf"))
        for i in range(6):        # untouched
            score.add(outcome(0.8, 0.8, 1, 1, file=f"b{i}.pdf"))
        table = goodhart(score.to_dict())
        assert table["cleared_and_damaged"] == 6
        text = render(compare({"reprompt": score}))
        assert "no benign reading" in text

    def test_lift_says_damage_and_clearing_travel_together(self):
        from eval.repair import goodhart

        score = RepairScore()
        for i in range(8):        # damaged, all cleared
            score.add(outcome(0.9, 0.3, 2, 0, file=f"a{i}.pdf"))
        for i in range(8):        # unchanged, none cleared
            score.add(outcome(0.8, 0.8, 2, 2, file=f"b{i}.pdf"))
        table = goodhart(score.to_dict())
        assert table["clear_rate_when_damaged"] == 1.0
        assert table["clear_rate_otherwise"] == 0.0
        assert table["lift"] is None       # undefined against a zero rate, not infinite

    def test_a_document_with_no_gates_cannot_clear_them(self):
        """gates_before == 0 means nothing was firing. Counting that as `cleared`
        would credit the loop for silence it did not produce."""
        from eval.repair import goodhart

        score = RepairScore()
        score.add(outcome(0.9, 0.3, gates_before=0, gates_after=0, file="a.pdf"))
        table = goodhart(score.to_dict())
        assert table["cleared_and_damaged"] == 0
        assert table["held_and_damaged"] == 1

    def test_a_report_without_per_document_gates_says_so(self):
        from eval.repair import goodhart

        assert goodhart({"deltas": {"a.pdf": -0.5}})["available"] is False


class TestInvariants:
    """Properties that must hold for any input, not just the cases I thought of.

    Written as explicit generators rather than with Hypothesis, which is not a
    dependency here. The point is the same: these assert structure, so they fail on
    inputs no example-based test would have covered.
    """

    def rows(self, seed, n=40):
        import random as rnd
        rng = rnd.Random(seed)
        out = []
        for i in range(n):
            before = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])
            after = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])
            out.append(outcome(before, after, rng.randint(0, 3), rng.randint(0, 3),
                               file=f"invoices/doc_{i}__{rng.choice(['fax','light'])}.pdf"))
        return out

    def arm_from(self, name, rows):
        score = RepairScore(arm=name)
        for row in rows:
            score.add(row)
        return score

    def test_an_unchanged_document_is_neither_improved_nor_damaged(self):
        for value in (0.0, 0.33, 0.5, 1.0):
            row = outcome(value, value)
            assert not row.improved and not row.damaged
            assert row.delta == 0

    def test_a_strictly_better_document_is_never_damaged(self):
        for seed in range(20):
            for row in self.rows(seed, 10):
                if row.after > row.before + 1e-9:
                    assert not row.damaged
                if row.after < row.before - 1e-9:
                    assert not row.improved

    def test_swapping_the_arms_negates_the_paired_mean(self):
        for seed in (1, 2, 3):
            a = self.arm_from("a", self.rows(seed)).to_dict()
            b = self.arm_from("b", self.rows(seed + 100)).to_dict()
            assert paired(a, b)["mean"] == pytest.approx(-paired(b, a)["mean"],
                                                         abs=1e-9)

    def test_an_arm_against_itself_is_exactly_zero(self):
        for seed in (4, 5, 6):
            a = self.arm_from("a", self.rows(seed)).to_dict()
            pair = paired(a, a)
            assert pair["mean"] == 0
            assert pair["interval"] == [0.0, 0.0]
            assert not pair["resolvable"]

    def test_document_order_cannot_change_the_result(self):
        import random as rnd

        rows = self.rows(7)
        shuffled = list(rows)
        rnd.Random(99).shuffle(shuffled)
        one = self.arm_from("a", rows).to_dict()
        two = self.arm_from("a", shuffled).to_dict()
        for key in ("net_delta", "improved", "damaged", "worst_delta",
                    "median_delta", "gates_clear", "net_delta_ci"):
            assert one[key] == two[key], key

    def test_duplicating_every_document_leaves_point_estimates_alone(self):
        """The interval may move because n changes. The point estimate must not."""
        rows = self.rows(8)
        one = self.arm_from("a", rows).to_dict()
        two = self.arm_from("a", rows + [
            outcome(r.before, r.after, r.gates_before, r.gates_after,
                    file=r.file.replace("invoices/", "copies/")) for r in rows
        ]).to_dict()
        assert one["net_delta"] == pytest.approx(two["net_delta"], abs=1e-9)
        assert one["damaged_rate"] == pytest.approx(two["damaged_rate"], abs=1e-9)
        assert two["damaged"] == 2 * one["damaged"]

    def test_the_bootstrap_is_deterministic(self):
        """An interval that moves when the report is re-rendered is not a
        measurement."""
        rows = self.rows(9)
        first = self.arm_from("a", rows).to_dict()["net_delta_ci"]
        second = self.arm_from("a", rows).to_dict()["net_delta_ci"]
        assert first == second

    def test_clustering_by_source_collapses_profiles_of_one_page(self):
        """Four degradations of one document are one cluster, not four observations."""
        from eval.repair import source_of

        assert source_of("forms/claim_5040__fax.pdf") == "forms/claim_5040"
        assert source_of("forms/claim_5040.pdf") == "forms/claim_5040"
        rows = [outcome(0.5, 0.9, file=f"forms/claim_1__{p}.pdf")
                for p in ("fax", "light", "photo", "clean")]
        assert len({r.source for r in rows}) == 1

    def test_a_wilson_interval_stays_inside_zero_and_one(self):
        from eval.repair import wilson

        for successes, total in ((0, 5), (5, 5), (1, 3), (8, 40), (0, 1)):
            low, high = wilson(successes, total)
            assert 0.0 <= low <= high <= 1.0

    def test_field_counts_decide_damage_when_present(self):
        """Field correctness is countable, so damage is an integer question and a
        float tolerance cannot be what answers it."""
        row = Outcome(file="a.pdf", before=0.5, after=0.5, gates_before=1,
                      gates_after=1, correct_before=11, correct_after=12, fields=24)
        assert row.field_gain == 1
        assert row.improved and not row.damaged
