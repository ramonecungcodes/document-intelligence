"""Whether repair helped, and the many ways that question is answered wrongly.

Phase 6 ends with a repair success rate. Almost every obvious definition of that number
is one a repair loop can satisfy without improving a single document, so the definition
is most of the work.

THE TRAPS THIS IS BUILT TO FAIL LOUDLY ON

*Scoring by whether the complaints stopped.* A document enters repair because a
validator found that subtotal plus tax does not equal total. If success means the rule
stops firing, the shortest path is blanking the fields the rule reads -- an empty tax
cannot fail an arithmetic check. Success is measured against the corpus labels and never
against the gates; `gates_clear` is reported beside it because the distance between the
two is the diagnostic for exactly this.

*Reporting only what improved.* Repair rewrites answers that were partly right, and some
come back worse. `damaged` sits beside `improved` everywhere, with a Wilson interval,
because a rate observed on forty documents is an estimate and reads like a fact.

*Confusing two different questions.* "Should repair run at all" and "is the guidance
worth anything beyond resampling" are different hypotheses with different controls, and
a loop can win the second while losing the first -- which is what this corpus does. So
`no_repair` is an explicit arm rather than an implied starting point, and both
comparisons are reported.

*Treating four degradations of one page as four observations.* `invoice_001__fax`,
`__light` and `__photo` share a source document; documents sharing a page design share
structure. Resampling them as independent narrows every interval, and it is the same
mistake, in statistics, that document-level holdout made in Phase 3 when it reported a
model as generalising that had memorised templates. Intervals here are a **cluster
bootstrap over source documents**, not a normal approximation over rows.

*Planning sample size from the effect you happened to observe.* Near zero that is
violently unstable -- three runs of one comparison gave -0.007, +0.011 and +0.008, and
the implied n swings by orders of magnitude. Sample size is planned against a declared
minimum effect worth caring about instead.

THE PRIMARY OUTCOME, DECLARED IN ADVANCE

There are a lot of legitimate numbers here, which is a lot of chances to find a
favourable one. So:

    PRIMARY      document-weighted delta, guided against no_repair
    SECONDARY    guided against blind rerun; damage rate; the Goodhart cross-tab
    EXPLORATORY  profile and document-type slices, per-gate behaviour

Only the primary decides whether repair worked.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

# Field correctness is countable -- `core.normalize.Comparison.match` is a bool -- so
# "did this document get better" is an integer comparison and needs no tolerance. This
# constant survives only for callers that build Outcomes from ratios alone.
EPSILON = 1e-3

# The smallest improvement worth a second model call, in field accuracy. Sample-size
# planning uses this rather than the observed effect, because an observed effect near
# zero makes the required n meaningless.
MIN_USEFUL_DELTA = 0.01
BOOTSTRAP_ROUNDS = 4000
BOOTSTRAP_SEED = 20260828


def _rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def source_of(relative_path: str) -> str:
    """The document a degraded page came from.

    `forms/onboarding_5003__photo.pdf` and `forms/onboarding_5003__fax.pdf` are two
    photographs of one document, not two documents. This is the unit resampling has to
    treat as independent.
    """
    stem = relative_path[:-4] if relative_path.endswith(".pdf") else relative_path
    return stem.split("__")[0]


# --------------------------------------------------------------------- intervals

def wilson(successes: int, total: int, z: float = 1.96):
    """A confidence interval for a proportion that behaves near 0 and 1.

    The textbook normal interval on a proportion runs out of bounds at the edges and is
    badly wrong for small counts -- and small counts is the case here, since a damage
    rate is often eight documents out of forty. Wilson is the standard fix and costs
    four lines.
    """
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5) / denominator
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def cluster_bootstrap(values_by_cluster: dict, rounds: int = BOOTSTRAP_ROUNDS,
                      seed: int = BOOTSTRAP_SEED):
    """Resample whole clusters, not rows, and return the interval of the mean.

    Two reasons this replaces `mean +/- 1.96 * stderr`.

    The distribution is wrong for a normal approximation: per-document differences here
    are bounded, semi-discrete, and overwhelmingly exactly zero -- 156 of 200 in the
    degraded run. A spike at zero with thin asymmetric tails is not what that formula
    assumes.

    And the rows are not independent. Four degradations of one page move together, so
    resampling rows pretends there is four times more information than there is. The
    unit resampled here is the source document, with all of its profiles carried along.

    Seeded, so the same data always yields the same interval. An interval that moves
    when the report is re-rendered is not a measurement.
    """
    # Sorted by cluster key, not left in insertion order. The resampler indexes into
    # this list, so an order that follows however the documents happened to arrive
    # makes the interval depend on the order of the input file -- the same data,
    # shuffled, produced [-0.1938, 0.1375] and [-0.1875, 0.1437]. Found by the
    # invariant test asserting that document order cannot change a result, which is
    # precisely the kind of bug no example-based test would have looked for.
    clusters = [values_by_cluster[key] for key in sorted(values_by_cluster)
                if values_by_cluster[key]]
    if len(clusters) < 3:
        return None
    rng = random.Random(seed)
    count = len(clusters)
    means = []
    for _ in range(rounds):
        total = 0.0
        size = 0
        for _ in range(count):
            picked = clusters[rng.randrange(count)]
            total += sum(picked)
            size += len(picked)
        means.append(total / size if size else 0.0)
    means.sort()
    low = means[int(0.025 * len(means))]
    high = means[min(len(means) - 1, int(0.975 * len(means)))]
    return [round(low, 4), round(high, 4)]


def documents_needed(values_by_cluster: dict, minimum: float = MIN_USEFUL_DELTA):
    """Roughly how many documents would reliably detect `minimum`, at 80% power.

    Deliberately not a function of the observed effect. Observed effects near zero make
    that number swing wildly between runs of the same experiment, which is how a
    sample-size estimate becomes noise dressed as a plan.

    Inflated by the mean cluster size, because documents inside a cluster are not
    independent and the effective sample is the number of sources.
    """
    flat = [value for rows in values_by_cluster.values() for value in rows]
    if len(flat) < 3 or not minimum:
        return None
    mean = sum(flat) / len(flat)
    variance = sum((v - mean) ** 2 for v in flat) / (len(flat) - 1)
    if not variance:
        return None
    # (z_alpha/2 + z_beta)^2 = (1.96 + 0.8416)^2, 5% two-sided at 80% power.
    per_source = 7.849 * variance / (minimum ** 2)
    inflation = len(flat) / max(1, len(values_by_cluster))
    return int(per_source * inflation) + 1


# ----------------------------------------------------------------------- outcomes

@dataclass
class Outcome:
    """One document through one repair arm."""

    file: str
    before: float                  # field accuracy as first extracted
    after: float                   # field accuracy after the arm ran
    gates_before: int
    gates_after: int
    attempts: int = 0
    doc_type: str = ""
    profile: str = "clean"
    error: str = ""                # the arm failed; the document keeps `before`
    # Raw counts when the caller has them. Field correctness is countable, so damage is
    # an integer question and does not need a float tolerance to answer.
    correct_before: int = None
    correct_after: int = None
    fields: int = None
    # What this document is not independent of.
    source: str = ""
    layout: object = None

    def __post_init__(self):
        if not self.source:
            self.source = source_of(self.file)

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def field_gain(self):
        """Fields gained, as an integer, when the counts are known."""
        if self.correct_before is None or self.correct_after is None:
            return None
        return self.correct_after - self.correct_before

    @property
    def improved(self) -> bool:
        gain = self.field_gain
        return gain > 0 if gain is not None else self.delta > EPSILON

    @property
    def damaged(self) -> bool:
        gain = self.field_gain
        return gain < 0 if gain is not None else self.delta < -EPSILON

    @property
    def gates_clear(self) -> bool:
        # A document with nothing firing cannot clear anything; counting it would
        # credit the loop for silence it did not produce.
        return self.gates_after == 0 and self.gates_before > 0


@dataclass
class RepairScore:
    """One arm: what it cost, what it fixed, and what it broke."""

    arm: str = "repair"
    outcomes: list = field(default_factory=list)

    def add(self, outcome: Outcome) -> None:
        self.outcomes.append(outcome)

    def __len__(self) -> int:
        return len(self.outcomes)

    def _clustered(self, pick) -> dict:
        out = defaultdict(list)
        for row in self.outcomes:
            out[row.source].append(pick(row))
        return dict(out)

    def to_dict(self) -> dict:
        rows = self.outcomes
        if not rows:
            return {"arm": self.arm, "documents": 0}
        improved = [r for r in rows if r.improved]
        damaged = [r for r in rows if r.damaged]
        ran = [r for r in rows if not r.error]
        deltas = sorted(r.delta for r in rows)

        before = sum(r.before for r in rows) / len(rows)
        after = sum(r.after for r in rows) / len(rows)

        # Document-weighted and field-weighted are different estimands and neither may
        # silently stand in for the other. A 9-field W-9 and a 24-field onboarding form
        # get one vote each in the first and 9 against 24 in the second, and the schemas
        # here span that whole range.
        graded = sum(r.fields for r in rows if r.fields)
        field_weighted = None
        if graded and all(r.correct_before is not None
                          and r.correct_after is not None for r in rows):
            gained = sum(r.correct_after - r.correct_before for r in rows)
            field_weighted = round(gained / graded, 4)

        return {
            "arm": self.arm,
            "documents": len(rows),
            "attempts": sum(r.attempts for r in rows),

            # Failures belong beside the quality figure, not folded into it. A loop
            # succeeding on 10% of documents and adding 0.20 when it does has an
            # all-document delta of +0.02 and is operationally useless.
            "failed": len(rows) - len(ran),
            # Kept under its old name too. Renaming a key in a report format is how a
            # reader that still works silently starts showing nothing.
            "errors": len(rows) - len(ran),
            "failure_rate": _rate(len(rows) - len(ran), len(rows)),

            "accuracy_before": round(before, 4),
            "accuracy_after": round(after, 4),
            # Net, because a loop is one decision -- run it or don't -- and what it is
            # worth is the sum of its help and its harm.
            "net_delta": round(after - before, 4),
            "net_delta_field_weighted": field_weighted,
            "net_delta_when_it_ran": (round(sum(r.delta for r in ran) / len(ran), 4)
                                      if ran else None),
            "net_delta_ci": cluster_bootstrap(self._clustered(lambda r: r.delta)),

            "improved": len(improved),
            "damaged": len(damaged),
            "unchanged": len(rows) - len(improved) - len(damaged),
            "improved_rate": _rate(len(improved), len(rows)),
            "damaged_rate": _rate(len(damaged), len(rows)),
            # A damage rate is an estimate. Eight of forty is 20% and could be 9% or
            # 36%, and only one of those is shippable.
            "damaged_rate_ci": wilson(len(damaged), len(rows)),
            "improved_rate_ci": wilson(len(improved), len(rows)),
            "gain_when_improved": (round(sum(r.delta for r in improved)
                                         / len(improved), 4) if improved else None),
            "loss_when_damaged": (round(sum(r.delta for r in damaged)
                                        / len(damaged), 4) if damaged else None),

            # The left tail. For an autonomous loop, "+0.02 mean, worst document -0.71"
            # and "+0.02 mean, worst document -0.04" are different products, and a mean
            # cannot tell them apart.
            "median_delta": round(deltas[len(deltas) // 2], 4),
            "p10_delta": round(deltas[max(0, int(0.10 * len(deltas)) - 1)], 4),
            "worst_delta": round(deltas[0], 4),

            "gates_clear": sum(1 for r in rows if r.gates_clear),
            "gates_clear_rate": _rate(sum(1 for r in rows if r.gates_clear), len(rows)),

            "sources": len({r.source for r in rows}),
            "deltas": {r.file: round(r.delta, 4) for r in rows},
            "gates": {r.file: [r.gates_before, r.gates_after] for r in rows},
            "clusters": {r.file: r.source for r in rows},
        }


def no_repair_arm(outcomes) -> "RepairScore":
    """The control: the original extraction, left alone.

    An explicit arm rather than an implied starting point, because "should repair run
    at all" is a different hypothesis from "is the guidance worth more than resampling",
    and a loop can win the second while losing the first. Making the control a real arm
    means both comparisons are computed the same way and neither can be quietly skipped.
    """
    score = RepairScore(arm="no_repair")
    for row in outcomes:
        score.add(Outcome(
            file=row.file, before=row.before, after=row.before,
            gates_before=row.gates_before, gates_after=row.gates_before,
            attempts=0, doc_type=row.doc_type, profile=row.profile,
            correct_before=row.correct_before, correct_after=row.correct_before,
            fields=row.fields, source=row.source, layout=row.layout))
    return score


# -------------------------------------------------------------------- comparison

def paired(a: dict, b: dict) -> dict:
    """Arm `a` against arm `b`, document by document, clustered by source.

    Paired, because both arms ran the identical document set from the identical starting
    extraction, so the unpaired difference of two means throws away the strongest thing
    known -- that a document hard for one arm was hard for the other.

    The interval is a cluster bootstrap. Four degradations of one page are not four
    observations, and treating them as such narrows every interval in the report.
    """
    shared = sorted(set(a.get("deltas") or {}) & set(b.get("deltas") or {}))
    if len(shared) < 3:
        return {"documents": len(shared), "mean": None, "interval": None,
                "resolvable": False, "clusters": 0, "needed": None,
                "better": 0, "worse": 0, "tied": 0}
    clusters = a.get("clusters") or {}
    by_cluster = defaultdict(list)
    for key in shared:
        by_cluster[clusters.get(key, key)].append(a["deltas"][key] - b["deltas"][key])

    diffs = [d for rows in by_cluster.values() for d in rows]
    mean = sum(diffs) / len(diffs)
    interval = cluster_bootstrap(dict(by_cluster))
    return {
        "documents": len(diffs),
        "clusters": len(by_cluster),
        "mean": round(mean, 4),
        "interval": interval,
        # The interval excluding zero is the only thing that licenses a claim. When it
        # does not, the difference has been observed once, not measured.
        "resolvable": bool(interval and (interval[0] > 0 or interval[1] < 0)),
        "needed": documents_needed(dict(by_cluster)),
        "better": sum(1 for d in diffs if d > EPSILON),
        "worse": sum(1 for d in diffs if d < -EPSILON),
        "tied": sum(1 for d in diffs if abs(d) <= EPSILON),
    }


def compare(arms: dict) -> dict:
    """Every arm against the two controls that matter.

    `no_repair` answers "should this run at all" and `rerun` answers "is the guidance
    worth more than another sample". Both are reported for every other arm, because a
    loop that beats one and loses the other is the interesting case and it is this
    corpus's actual result.
    """
    scored = {name: score.to_dict() for name, score in arms.items()}
    control = scored.get("no_repair")

    def baseline_for(name):
        """The blind arm at the SAME call budget.

        Budgeted runs name their arms `reprompt@2`, so a lookup for a bare "rerun"
        finds nothing and the report quietly loses its baseline -- which it did, and
        printed "baseline: none" while two perfectly good blind arms sat in the same
        dict. Comparing `reprompt@3` against `rerun@1` would have been worse than
        losing it: that prices two extra samples as though they were the guidance.
        """
        if "@" in name:
            return scored.get("rerun@" + name.split("@", 1)[1])
        return scored.get("rerun")

    any_baseline = any(n == "rerun" or n.startswith("rerun@") for n in scored)
    pairs, absolute = {}, {}
    for name, row in scored.items():
        row["over_rerun"] = None
        if not row.get("documents"):
            continue
        baseline = baseline_for(name)
        blind = name == "rerun" or name.startswith("rerun@")
        if baseline and not blind and name != "no_repair":
            row["over_rerun"] = round((row["net_delta"] or 0)
                                      - (baseline["net_delta"] or 0), 4)
            pairs[name] = paired(row, baseline)
        if control and name != "no_repair":
            absolute[name] = paired(row, control)

    return {"arms": scored,
            "baseline": "rerun" if any_baseline else None,
            "control": "no_repair" if control else None,
            "paired": pairs,
            "absolute": absolute,
            "documents": max((r.get("documents", 0) for r in scored.values()),
                             default=0)}


def goodhart(arm: dict) -> dict:
    """Did the gates go quiet on the same documents that got worse?

    The aggregate -- 52 cleared, 52 damaged -- is suggestive and proves nothing. Two
    disjoint groups of 52 give identical totals and mean something entirely different.
    Only the per-document join separates "silencing the critics" from "two unrelated
    things happened".

    The two conditional probabilities are reported before the ratio, because "55% of
    damaged documents cleared their gates against 12% of the rest" is far harder to
    misread than "4.4x", and the ratio carries a wide interval that a bare multiple
    hides entirely.
    """
    deltas = arm.get("deltas") or {}
    gates = arm.get("gates") or {}
    shared = sorted(set(deltas) & set(gates))
    if not shared:
        return {"documents": 0, "available": False}

    cells = {"cleared_and_damaged": 0, "cleared_and_improved": 0,
             "cleared_and_unchanged": 0, "held_and_damaged": 0,
             "held_and_improved": 0, "held_and_unchanged": 0}
    cleared_damaged = cleared_ok = damaged = intact = 0
    for key in shared:
        delta = deltas[key]
        before, after = gates[key]
        cleared = before > 0 and after == 0
        state = ("damaged" if delta < -EPSILON
                 else "improved" if delta > EPSILON else "unchanged")
        cells[f"{'cleared' if cleared else 'held'}_and_{state}"] += 1
        if state == "damaged":
            damaged += 1
            cleared_damaged += cleared
        else:
            intact += 1
            cleared_ok += cleared

    rate_damaged = cleared_damaged / damaged if damaged else None
    rate_intact = cleared_ok / intact if intact else None
    ratio_ci = None
    if rate_damaged and rate_intact:
        # Katz log interval on the risk ratio. A bare multiple reads as precise; on
        # counts this size it is anything but.
        log_rr = math.log(rate_damaged / rate_intact)
        se = math.sqrt(max(0.0, (1 - rate_damaged) / cleared_damaged
                           + (1 - rate_intact) / max(1, cleared_ok)))
        ratio_ci = [round(math.exp(log_rr - 1.96 * se), 3),
                    round(math.exp(log_rr + 1.96 * se), 3)]

    return {
        "documents": len(shared), "available": True, **cells,
        "damaged": damaged, "not_damaged": intact,
        "clear_rate_when_damaged": (round(rate_damaged, 4)
                                    if rate_damaged is not None else None),
        "clear_rate_when_damaged_ci": wilson(cleared_damaged, damaged),
        "clear_rate_otherwise": (round(rate_intact, 4)
                                 if rate_intact is not None else None),
        "clear_rate_otherwise_ci": wilson(cleared_ok, intact),
        "lift": (round(rate_damaged / rate_intact, 3)
                 if rate_damaged and rate_intact else None),
        "lift_ci": ratio_ci,
    }


def budget_curve(arms: dict, names, budget: int) -> dict:
    """Each arm at each call budget, and the arms against each other at matched ones.

    The comparison only means anything at equal budgets. Three guided attempts against
    one blind sample measures the extra sampling as though it were the guidance, which
    is the single easiest way to make a repair loop look like it works.

    Two shapes worth reading off it.

    Whether the guided arm *slopes*. The blind arm is expected to be flat: without a
    selector, the third independent sample is no better than the first, and the only
    selector available is the validators -- picking whichever attempt satisfies them is
    the optimisation this module exists to catch. So a flat blind line is the correct
    null and any slope in the guided one is what iteration bought.

    Whether damage compounds. A guided attempt that made the record worse hands that
    worse record to the next attempt. If the damage rate climbs with budget while the
    net delta does not, the loop is not converging on the document, it is drifting away
    from it, and the third call is buying harm.
    """
    steps = list(range(1, budget + 1))
    out = {"budget": budget, "arms": {}, "matched": {}}
    for name in names:
        row = []
        for step in steps:
            arm = arms.get(f"{name}@{step}") or arms.get(name)
            if arm is None:
                continue
            data = arm.to_dict() if isinstance(arm, RepairScore) else arm
            row.append({
                "attempts": step,
                "net_delta": data.get("net_delta"),
                "net_delta_ci": data.get("net_delta_ci"),
                "improved": data.get("improved"),
                "damaged": data.get("damaged"),
                "damaged_rate": data.get("damaged_rate"),
                "damaged_rate_ci": data.get("damaged_rate_ci"),
                "worst_delta": data.get("worst_delta"),
                "gates_clear": data.get("gates_clear"),
            })
        out["arms"][name] = row

    control = arms.get("no_repair")
    control = (control.to_dict() if isinstance(control, RepairScore) else control)
    guided = [n for n in names if n != "rerun"]
    for name in guided:
        row = []
        for step in steps:
            a = arms.get(f"{name}@{step}") or arms.get(name)
            b = arms.get(f"rerun@{step}") or arms.get("rerun")
            if a is None or b is None:
                continue
            a = a.to_dict() if isinstance(a, RepairScore) else a
            b = b.to_dict() if isinstance(b, RepairScore) else b
            entry = {"attempts": step, "vs_rerun": paired(a, b)}
            if control:
                entry["vs_no_repair"] = paired(a, control)
            row.append(entry)
        out["matched"][name] = row
    return out


def render_budget_curve(curve: dict) -> str:
    if not curve or not curve.get("arms"):
        return ""
    out = ["", "BUDGET CURVE  -  arms compared only at equal call counts", ""]
    out.append(f"  {'arm':<12}{'calls':>7}{'net':>10}{'95% interval':>24}"
               f"{'better':>8}{'worse':>7}{'damage':>9}{'worst':>9}")
    for name, rows in curve["arms"].items():
        for row in rows:
            ci = row.get("net_delta_ci")
            band = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "--"
            out.append(f"  {name:<12}{row['attempts']:>7}"
                       f"{_fmt(row['net_delta'], 10, 4, sign=True)}{band:>24}"
                       f"{row['improved']:>8}{row['damaged']:>7}"
                       f"{(row['damaged_rate'] or 0):>9.1%}"
                       f"{_fmt(row['worst_delta'], 9, sign=True)}")
    out.append("  net is against the original extraction, so every row answers")
    out.append("  \"was this call worth making\" rather than \"did anything change\".")

    for name, rows in curve["matched"].items():
        if not rows:
            continue
        out.append("")
        out.append(f"  {name} against the blind re-run, at matched budgets")
        for row in rows:
            pair = row["vs_rerun"]
            if pair.get("mean") is None:
                continue
            band = _interval(pair)
            verdict = "resolvable" if pair["resolvable"] else "spans zero"
            out.append(f"    {row['attempts']} call(s)  "
                       f"{_fmt(pair['mean'], 9, 4, sign=True)}   {band:>22}   "
                       f"{verdict}")

    # The two readings the curve exists for, stated rather than left to the eye.
    for name, rows in curve["arms"].items():
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]

        def separated(low_row, high_row, key):
            """Do the two intervals fail to overlap?

            A bare comparison of point estimates fired this warning on a 0.7 point
            move in the damage rate -- one document -- whose interval overlapped the
            first almost entirely. That is precisely the error the rest of this module
            refuses to make, committed by the module itself. A difference between two
            budgets is only worth naming when their intervals are disjoint.
            """
            a, b = low_row.get(key), high_row.get(key)
            if not a or not b:
                return False
            return b[0] > a[1] or a[0] > b[1]

        if ((last["net_delta"] or 0) < (first["net_delta"] or 0)
                and separated(first, last, "net_delta_ci")):
            out.append("")
            out.append(f"  {name} is WORSE at {last['attempts']} calls than at "
                       f"{first['attempts']}: "
                       f"{first['net_delta']:+.4f} -> {last['net_delta']:+.4f}, "
                       f"intervals disjoint.")
            out.append("  Extra attempts are buying harm. Cap the budget lower.")
        if ((last["damaged_rate"] or 0) > (first["damaged_rate"] or 0)
                and separated(first, last, "damaged_rate_ci")):
            out.append(f"  {name} damage rate climbs with budget: "
                       f"{(first['damaged_rate'] or 0):.1%} -> "
                       f"{(last['damaged_rate'] or 0):.1%}, intervals disjoint. An "
                       f"attempt that made the record worse")
            out.append("  is handing that worse record to the next one.")

        # Flat is a finding too, and the more likely one. Without saying so, a reader
        # sees three rows that differ in the fourth decimal and infers a trend.
        if not separated(first, last, "net_delta_ci"):
            out.append("")
            out.append(f"  {name}: budget {first['attempts']} to {last['attempts']} is "
                       f"flat within noise "
                       f"({first['net_delta']:+.4f} -> {last['net_delta']:+.4f}, "
                       f"intervals overlap).")
            out.append("  The extra calls bought nothing measurable either way.")
    return "\n".join(out)


# --------------------------------------------------------------- transitions
# What repair did to each field, rather than what it did to each document.
#
# A document-level delta hides the mechanism completely. Consider a repair that takes
# a four-field document from 2/4 to 3/4:
#
#     vendor_name      right  ->  right
#     invoice_number   right  ->  right
#     service_address  missed ->  fabricated      <- the document has no such field
#     total            wrong  ->  right
#
# That is +0.25 and is scored as an improvement. Operationally it did two unrelated
# things: it repaired a total, and it invented an address the page does not contain.
# The second is the expensive error in this project's terms -- a blank field gets
# looked at and a confident wrong one flows downstream -- and the document-level
# number cannot see it at all.
#
# Ordered worst-first, because the ones that matter are the ones a mean hides.
TRANSITIONS = (
    ("right_to_wrong", "a correct value was replaced with a wrong one"),
    ("right_to_missed", "a correct value was dropped"),
    ("right_to_fabricated", "a correct value became one the page does not carry"),
    ("missed_to_fabricated", "an empty field was filled with something invented"),
    ("correct_blank_to_fabricated", "a correctly empty field was invented into"),
    ("wrong_to_wrong", "still wrong, differently"),
    ("wrong_to_fabricated", "wrong, and now for a field the page does not carry"),
    ("wrong_to_missed", "a wrong value was dropped"),
    ("missed_to_missed", "still empty"),
    ("fabricated_to_fabricated", "still invented"),
    ("fabricated_to_missed", "an invented value was withdrawn"),
    ("fabricated_to_right", "an invented value became the right one"),
    ("missed_to_right", "an empty field was correctly filled"),
    ("wrong_to_right", "a wrong value was corrected"),
    ("right_to_right", "unchanged and correct"),
    ("correct_blank_to_correct_blank", "unchanged and correctly empty"),
)
# Which transitions are damage, which are repair. Everything else is neutral -- and
# `wrong_to_wrong` is deliberately neutral rather than damage, because a value that was
# already wrong being wrong differently costs nothing new.
DAMAGING = {"right_to_wrong", "right_to_missed", "right_to_fabricated",
            "missed_to_fabricated", "correct_blank_to_fabricated",
            "wrong_to_fabricated"}
# Accuracy actually moved: the field is correct now and was not before.
REPAIRING = {"wrong_to_right", "missed_to_right", "fabricated_to_right"}
# Neither. The field is still not correct, but a silent wrong value became a visible
# gap -- which this project has argued since Phase 1 is the cheaper failure, because a
# blank gets looked at and a confident wrong one flows downstream unchallenged.
#
# Counted apart rather than as repair, because it was counted as repair first and the
# smoke run reported "repaired 6 fields" for six values that were dropped and never
# corrected. A category that flatters the loop is the one thing this module cannot
# have.
SAFER = {"wrong_to_missed", "fabricated_to_missed"}

_SHORT = {"right": "right", "correctly_blank": "correct_blank",
          "missed": "missed", "fabricated": "fabricated", "wrong": "wrong"}


def transition_of(before: str, after: str) -> str:
    return f"{_SHORT.get(before, before)}_to_{_SHORT.get(after, after)}"


class Transitions:
    """Field-level movement, counted across a whole arm.

    Weighted counts are kept beside unweighted ones rather than replacing them. The
    unweighted figure is the scientifically clean one; the weighted figure says what it
    costs to break an invoice total against a vendor phone number. A single number
    claiming to be both is neither.
    """

    def __init__(self):
        self.counts = defaultdict(int)
        self.weighted = defaultdict(float)
        self.by_field = defaultdict(lambda: defaultdict(int))
        self.documents = 0

    def add(self, before: dict, after: dict, weights: dict = None) -> None:
        self.documents += 1
        weights = weights or {}
        for name, was in before.items():
            now = after.get(name)
            if now is None:
                continue                     # field not graded on the second pass
            key = transition_of(was, now)
            self.counts[key] += 1
            self.weighted[key] += weights.get(name, 1.0)
            if key in DAMAGING or key in REPAIRING or key in SAFER:
                self.by_field[name][key] += 1

    def to_dict(self) -> dict:
        damaged = sum(self.counts[k] for k in DAMAGING)
        repaired = sum(self.counts[k] for k in REPAIRING)
        safer = sum(self.counts[k] for k in SAFER)
        w_damaged = sum(self.weighted[k] for k in DAMAGING)
        w_repaired = sum(self.weighted[k] for k in REPAIRING)
        rows = [{"transition": name, "help": note, "count": self.counts[name],
                 "weighted": round(self.weighted[name], 2)}
                for name, note in TRANSITIONS if self.counts[name]]
        worst = sorted(
            ({"field": field,
              "damaged": sum(v for k, v in moves.items() if k in DAMAGING),
              "repaired": sum(v for k, v in moves.items() if k in REPAIRING),
              "moves": dict(moves)}
             for field, moves in self.by_field.items()),
            key=lambda r: r["damaged"] - r["repaired"], reverse=True)
        return {
            "documents": self.documents,
            "fields_repaired": repaired,
            "fields_damaged": damaged,
            # Still wrong, but wrong in a way a person will notice.
            "fields_made_visible": safer,
            "net_fields": repaired - damaged,
            "fields_repaired_weighted": round(w_repaired, 2),
            "fields_damaged_weighted": round(w_damaged, 2),
            "net_fields_weighted": round(w_repaired - w_damaged, 2),
            # Called out on its own because it is the transition this project has
            # argued about since Phase 1: a blank a person would have caught, turned
            # into a confident value nobody will.
            "invented": (self.counts["missed_to_fabricated"]
                         + self.counts["correct_blank_to_fabricated"]
                         + self.counts["right_to_fabricated"]
                         + self.counts["wrong_to_fabricated"]),
            "rows": rows,
            "worst_fields": worst[:10],
        }


def render_transitions(data: dict, arm: str = "") -> str:
    if not data or not data.get("rows"):
        return ""
    out = ["", f"FIELD TRANSITIONS{f'  -  {arm}' if arm else ''}", ""]
    out.append(f"  {'transition':<32}{'fields':>8}{'weighted':>10}   what it means")
    for row in data["rows"]:
        mark = ("!" if row["transition"] in DAMAGING
                else "+" if row["transition"] in REPAIRING
                else "~" if row["transition"] in SAFER else " ")
        out.append(f"  {mark} {row['transition']:<30}{row['count']:>8}"
                   f"{row['weighted']:>10.1f}   {row['help']}")
    out.append("")
    out.append(f"  repaired {data['fields_repaired']:>5} fields "
               f"({data['fields_repaired_weighted']:.1f} weighted)")
    out.append(f"  damaged  {data['fields_damaged']:>5} fields "
               f"({data['fields_damaged_weighted']:.1f} weighted)")
    out.append(f"  net      {data['net_fields']:>+5} fields "
               f"({data['net_fields_weighted']:+.1f} weighted)")
    if data.get("fields_made_visible"):
        out.append(f"  ~ {data['fields_made_visible']} more fields went from a wrong "
                   f"value to an empty one: still not")
        out.append("    correct, but a gap a reviewer sees rather than a value they "
                   "trust.")
    if data["invented"]:
        out.append("")
        out.append(f"  {data['invented']} fields were filled with values the page does "
                   f"not carry.")
        out.append("  A blank field gets looked at. A confident wrong one does not.")
    if data["net_fields"] > 0 and data["net_fields_weighted"] < 0:
        out.append("")
        out.append("  Positive unweighted, negative weighted: this repair traded "
                   "important fields")
        out.append("  for unimportant ones. The unweighted number is the one that "
                   "misleads here.")
    if data.get("worst_fields"):
        out.append("")
        out.append(f"  {'field':<28}{'damaged':>9}{'repaired':>10}")
        for row in data["worst_fields"]:
            if row["damaged"] or row["repaired"]:
                out.append(f"  {row['field'][:27]:<28}{row['damaged']:>9}"
                           f"{row['repaired']:>10}")
    return "\n".join(out)


def by_slice(score: RepairScore, key: str = "doc_type") -> list:
    """Per type or per profile. Exploratory only -- see the declared hierarchy."""
    groups = defaultdict(list)
    for row in score.outcomes:
        groups[getattr(row, key, "") or "unknown"].append(row)
    out = []
    for name, rows in sorted(groups.items()):
        improved = sum(1 for r in rows if r.improved)
        damaged = sum(1 for r in rows if r.damaged)
        out.append({
            key: name,
            "documents": len(rows),
            "net_delta": round(sum(r.delta for r in rows) / len(rows), 4),
            "improved": improved,
            "damaged": damaged,
        })
    return out


# ------------------------------------------------------------------------ render

def _fmt(value, width=8, digits=3, sign=False):
    if value is None:
        return f"{'--':>{width}}"
    return f"{value:>{'+' if sign else ''}{width}.{digits}f}"


def _interval(pair) -> str:
    if not pair or not pair.get("interval"):
        return "--"
    low, high = pair["interval"]
    return f"[{low:+.4f}, {high:+.4f}]"


def render(data: dict, slices: dict = None) -> str:
    out = ["", "REPAIR  -  did the second attempt help", ""]
    out.append(f"  documents            {data.get('documents', 0):>8}")
    arms = data.get("arms") or {}
    if data.get("control"):
        out.append("  control              no_repair, the original extraction")
    else:
        out.append("  control                  none   <- no no_repair arm, so nothing")
        out.append("     below answers whether repair should run at all")
    if data.get("baseline"):
        out.append("  baseline             blind re-run, same prompt, no feedback")
    else:
        out.append("  baseline                 none   <- no blind re-run arm, so any")
        out.append("     gain below includes what a second sample is worth")

    if not any(r.get("documents") for r in arms.values()):
        out.append("")
        out.append("  no documents.")
        return "\n".join(out)

    out.append("")
    out.append(f"  {'arm':<12}{'calls':>6}{'fail':>6}{'net':>9}{'field-wt':>10}"
               f"{'better':>8}{'worse':>7}{'worst':>9}{'gates':>7}")
    for name, row in arms.items():
        if not row.get("documents"):
            continue
        out.append(
            f"  {name:<12}{row['attempts']:>6}{row['failed']:>6}"
            f"{_fmt(row['net_delta'], 9, sign=True)}"
            f"{_fmt(row.get('net_delta_field_weighted'), 10, sign=True)}"
            f"{row['improved']:>8}{row['damaged']:>7}"
            f"{_fmt(row['worst_delta'], 9, sign=True)}{row['gates_clear']:>7}")
    out.append("  net is document-weighted; field-wt weights by fields graded. They")
    out.append("  answer different questions and neither stands in for the other.")

    harmful = [n for n, r in arms.items()
               if r.get("documents") and (r.get("net_delta") or 0) < 0
               and n != "no_repair"]
    if harmful:
        out.append("")
        out.append(f"  NET-NEGATIVE: {', '.join(harmful)} left documents worse than "
                   f"they were found.")
        out.append("  Repair is optional. An arm that scores below zero should be off.")

    out.append("")
    out.append("  PRIMARY  -  should repair run at all (against no_repair)")
    if not data.get("absolute"):
        out.append("    not computed: no control arm.")
    for name, pair in (data.get("absolute") or {}).items():
        if pair["mean"] is None:
            continue
        verdict = ("worse than doing nothing" if pair["mean"] < 0
                   else "better than doing nothing")
        settled = "" if pair["resolvable"] else "  (interval spans zero)"
        out.append(f"    {name:<12}{_fmt(pair['mean'], 9, 4, sign=True)}   "
                   f"{_interval(pair)}   {verdict}{settled}")
    out.append("    clustered by source document, so four degradations of one page")
    out.append("    count once rather than four times")

    out.append("")
    out.append("  SECONDARY  -  is the guidance worth more than resampling "
               "(against rerun)")
    for name, pair in (data.get("paired") or {}).items():
        if pair["mean"] is None:
            out.append(f"    {name}: too few shared documents to compare.")
            continue
        out.append(f"    {name:<12}{_fmt(pair['mean'], 9, 4, sign=True)}   "
                   f"{_interval(pair)}   "
                   f"{pair['clusters']} sources, {pair['documents']} documents")
        out.append(f"      better on {pair['better']}, worse on {pair['worse']}, "
                   f"tied on {pair['tied']}")
        if not pair["resolvable"]:
            need = pair.get("needed")
            # Kept on one line: the phrase is what a reader greps for, and splitting
            # it across a wrap is how an assertion about it silently stops matching.
            out.append("      The interval does not exclude zero -- observed once,")
            out.append("      not measured, so do not quote the point estimate.")
            out.append(f"      Detecting a {MIN_USEFUL_DELTA:+.2f} effect needs about "
                       f"{need if need else 'many more'} documents.")
        elif name in harmful:
            # Only when THIS arm is itself harmful. It fired on any harmful arm in the
            # report, so a positive guided arm beating a negative blind one printed
            # "both arms are net-negative" -- a false sentence, in the block whose
            # entire job is to stop a true number being read as a good one.
            out.append("      Resolvable, but this arm is net-negative against doing")
            out.append("      nothing. It is less harmful, not helpful.")
            out.append("      It should not run on these documents.")
        elif harmful:
            out.append("      Resolvable, and this arm is not itself net-negative --")
            out.append(f"      only {', '.join(harmful)} is. The guidance is what")
            out.append("      separates them.")
        else:
            out.append("      The feedback is worth something beyond a second sample.")

    out.append("")
    out.append("  DAMAGE")
    out.append(f"  {'arm':<12}{'damaged':>9}{'rate':>8}{'95% interval':>22}"
               f"{'median':>9}{'p10':>9}")
    for name, row in arms.items():
        if not row.get("documents") or name == "no_repair":
            continue
        ci = row.get("damaged_rate_ci")
        band = f"[{ci[0]:.1%}, {ci[1]:.1%}]" if ci else "--"
        out.append(f"  {name:<12}{row['damaged']:>9}"
                   f"{(row['damaged_rate'] or 0):>8.1%}{band:>22}"
                   f"{_fmt(row['median_delta'], 9, sign=True)}"
                   f"{_fmt(row['p10_delta'], 9, sign=True)}")

    for name, row in arms.items():
        table = goodhart(row)
        if not table.get("available") or name == "no_repair":
            continue
        cad = table["cleared_and_damaged"]
        if not cad and not table["cleared_and_improved"]:
            continue
        out.append("")
        out.append(f"  GATES AGAINST TRUTH  -  {name}")
        out.append(f"  {'':<22}{'damaged':>10}{'improved':>10}{'unchanged':>11}")
        out.append(f"  {'gates cleared':<22}{cad:>10}"
                   f"{table['cleared_and_improved']:>10}"
                   f"{table['cleared_and_unchanged']:>11}")
        out.append(f"  {'gates still firing':<22}{table['held_and_damaged']:>10}"
                   f"{table['held_and_improved']:>10}"
                   f"{table['held_and_unchanged']:>11}")
        if (table["clear_rate_when_damaged"] is not None
                and table["clear_rate_otherwise"] is not None):
            hurt = table.get("clear_rate_when_damaged_ci") or [0, 0]
            other = table.get("clear_rate_otherwise_ci") or [0, 0]
            out.append(f"  {table['clear_rate_when_damaged']:.1%} of damaged documents "
                       f"cleared their gates   [{hurt[0]:.1%}, {hurt[1]:.1%}]")
            out.append(f"  {table['clear_rate_otherwise']:.1%} of the rest did"
                       f"                         [{other[0]:.1%}, {other[1]:.1%}]")
        if table.get("lift"):
            band = (f"  [{table['lift_ci'][0]}x, {table['lift_ci'][1]}x]"
                    if table.get("lift_ci") else "")
            out.append(f"  risk ratio {table['lift']}x{band}")
        if cad:
            out.append(f"  {cad} documents stopped tripping every rule AND came back "
                       f"further from the truth.")
            out.append("  There is no benign reading of that cell.")

    for name, row in arms.items():
        if (row.get("documents") and name != "no_repair"
                and row["damaged"] > row["improved"]):
            out.append("")
            out.append(f"  {name} damaged more documents than it improved "
                       f"({row['damaged']} against {row['improved']}).")

    for key, rows in (slices or {}).items():
        out.append("")
        out.append(f"  BY {key.replace('_', ' ').upper()}   (exploratory)")
        out.append(f"  {key:<24}{'n':>6}{'net':>9}{'better':>8}{'worse':>7}")
        for row in rows:
            out.append(f"  {str(row[key])[:23]:<24}{row['documents']:>6}"
                       f"{_fmt(row['net_delta'], 9, sign=True)}"
                       f"{row['improved']:>8}{row['damaged']:>7}")
    return "\n".join(out)
