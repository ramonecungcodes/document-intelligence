#!/usr/bin/env python3
"""Run repair arms over the documents the router flagged, and score what they did.

    python -m repair.cli run --predictions reports/v1-predicted-type.jsonl \\
        --corpus data --arms rerun,reprompt --limit 40

Every arm sees the identical set of documents, chosen once before any of them runs. That
is the point of the command: `rerun` and `reprompt` are the same request differing only
in whether the complaints are included, so anything that separates them is what the
feedback was worth. Selecting documents per arm, or running one arm on a different day
than the other, would price the corpus instead.

Scoring is against the corpus labels through `eval.score.per_document`, never against
whether the validators went quiet. See `eval/repair.py` for why that distinction is the
whole measurement.

Repair costs a model call per document per arm, so `--limit` exists and the default is
to touch nothing until it is set: two arms over a thousand documents is two thousand
calls, and that should be a decision rather than a default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from core import config as config_mod
from core import doctypes
from eval import repair as scoring_repair
from eval import score as scoring

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "/data")
REPORTS_DIR = os.environ.get("DI_REPORTS_DIR", "/reports")


def _candidates(args, config):
    """The documents to repair, and their complaints -- chosen once for every arm.

    The signals come from `eval.cli.signal_rows`, which is also what `route.cli` uses.
    They were computed separately here once, and the two disagreed: this module built
    features without the signals sidecar, so `classifier_confidence` was always absent,
    that gate never fired, and repair silently saw 40 flagged documents where the router
    saw 63. Neither number looked wrong. Two implementations of one decision is the
    failure this project keeps finding, and the fix is always the same -- have one.
    """
    import repair
    from eval.cli import signal_rows
    from route import policy as policy_mod
    from validate.base import build_all

    validators = build_all(config)
    policy = policy_mod.build(config)
    rows = signal_rows(args)
    graded = scoring.per_document(
        args.corpus, scoring.load_records(args.predictions),
        [t.strip() for t in args.only.split(",") if t.strip()] or None)

    by_file, truth_of = {}, {}
    for stem, labels in scoring.load_corpus(args.corpus, [
            t.strip() for t in args.only.split(",") if t.strip()] or None).items():
        spec = doctypes.for_label_file(stem)
        for label in labels:
            key = str(label.get("file", "")).replace("\\", "/")
            if key:
                by_file[key] = (spec, spec.variant_of(label) if spec else "")
                # The label itself, so a field's state before and after can be compared
                # against the same truth rather than against two readings of it.
                truth_of[key] = label

    records = {str(r.get("file", "")).replace("\\", "/"): r
               for r in scoring.load_records(args.predictions)}

    out = []
    for row in rows:
        key = row["file"]
        spec, variant = by_file.get(key, (None, ""))
        record = records.get(key)
        if spec is None or record is None:
            continue
        decision = policy.decide(row["signals"])
        if args.flagged_only and not decision.review:
            continue
        graded_before = graded.get(key) or {}
        out.append({
            "truth": truth_of.get(key),
            "file": key, "doc_type": row["truth"], "spec": spec, "variant": variant,
            "profile": row["profile"],
            "record": record,
            "before": row["outcome"],
            "correct_before": graded_before.get("fields_correct"),
            "fields": graded_before.get("fields_graded"),
            "layout": graded_before.get("layout"),
            "gates_before": len(decision.reasons),
            "complaints": repair.complaints_for(record, spec, variant, validators,
                                                decision),
        })
    if args.limit:
        out = out[:args.limit]
    return out, validators, policy


def _normalizer_for(config, name, corpus):
    from normalize.base import NORMALIZERS, build

    chosen = (config.chosen("normalizer", name) or "native").strip().lower()
    declares = {s.name for s in NORMALIZERS.get(
        chosen, type("x", (), {"SETTINGS": ()})).SETTINGS}
    return build(config=config, plugin=name,
                 overrides={"corpus": corpus} if "corpus" in declares else None)


def _merge(original: dict, produced: dict, spec, variant: str) -> dict:
    """The repaired record, keeping the harness's own keys and the file identity.

    Only the model's fields are replaced. A repair that dropped `file` would silently
    fail to join against the corpus and be scored as a missing document rather than a
    bad one.
    """
    from extract.runner import collapse_optional

    produced = dict(produced)
    # Optional fields are asked as a decision, not a slot: the model answers
    # {"status": ..., "value": ...} and the runner flattens it before anything else
    # sees it. This did not, so a repaired record kept the dict -- and the scorer,
    # comparing a dict against a blank truth, read it as a fabricated value.
    #
    # It cost the headline of Phase 6. The report said repair invented business_name
    # on 48 of 48 eligible W-9s; the model had actually answered "unclear" on 47 of
    # them, which collapses to absent, which is correct. All 53 of the guided arm's
    # inventions were this line missing.
    collapse_optional(produced, spec, variant)
    merged = {k: v for k, v in original.items()
              if k.startswith("_") or k == "file"}
    merged["doc_type"] = spec.name
    merged.update(produced)
    if variant and spec.variant_key:
        merged[spec.variant_key] = variant
    return merged


def run_arm(name, rows, config, args, validators, policy, budget: int = 1):
    """One arm over every candidate, scored after every attempt.

    Returns {attempts_used: RepairScore}, one entry per budget from 1 to `budget`, so
    the curve is measured rather than interpolated between its endpoints. The two arms
    are run at the same budgets over the same documents; otherwise a comparison prices
    three chances to sample a better answer against one, rather than pricing the
    guidance.

    Guided arms iterate -- attempt N sees attempt N-1's record and the complaints
    recomputed against it. Blind arms repeat the identical original request. Which one
    a repairer is comes from its ITERATIVE flag, not from anything here.
    """
    import repair
    from route import features as feature_mod
    from extract import backends

    repairer = repair.build(name, config, overrides={"max_attempts": 1})
    iterative = getattr(type(repairer), "ITERATIVE", False)
    print(f"  arm {name}: {repairer.describe()}  "
          f"({'iterative' if iterative else 'independent samples'}), "
          f"budgets 1..{budget}")
    backend = backends.build(config=config, plugin=args.extractor)
    normalizer = _normalizer_for(config, args.normalizer, args.corpus)
    print(f"  normalizer: {normalizer.describe()}")

    # Repair must re-read the page the way the original extraction did. Reading it with
    # a better engine would show up as a repair gain that a prompt had nothing to do
    # with -- the single most attributable-looking wrong number this stage can produce.
    was = {str((row["record"].get("_normalizer") or {}).get("engine") or "?")
           for row in rows}
    now = normalizer.describe().split()[0]
    if was and not any(now.split(":")[0] in engine for engine in was):
        print(f"  WARNING: these predictions were read with {sorted(was)} and repair "
              f"is reading with {now!r}. Any gain below may be the reader, not the "
              f"repair.", file=sys.stderr)

    # records[k] is every document's answer after k attempts.
    records = {k: [] for k in range(1, budget + 1)}
    failures = {k: [] for k in range(1, budget + 1)}
    for index, row in enumerate(rows, 1):
        path = os.path.join(args.corpus, row["file"])
        page = normalizer.read(path)
        current = row["record"]
        complaints = row["complaints"]
        broke = ""
        for step in range(1, budget + 1):
            context = repair.Context(
                backend=backend, doctype=row["spec"], variant=row["variant"],
                path=path, relative_path=row["file"],
                # An iterative arm argues with its own latest answer; an independent
                # one is handed the original every time, which is what makes its curve
                # a sampling control rather than a second iterative arm.
                record=current if iterative else row["record"],
                text=page.text,
                complaints=complaints if iterative else row["complaints"],
                normalizer=normalizer)
            result = repairer.repair(context)
            if result.changed:
                produced = _merge(row["record"], result.record, row["spec"],
                                  row["variant"])
                if iterative:
                    current = produced
                    # Recomputed against the new answer, never carried over. A prompt
                    # complaining about a value the record no longer holds is asking
                    # the model to fix something that is not there.
                    signals = feature_mod.extract(current, row["spec"], row["variant"],
                                                  validators)
                    complaints = repair.complaints_for(
                        current, row["spec"], row["variant"], validators,
                        policy.decide(signals))
            else:
                produced = dict(current if iterative else row["record"])
                broke = broke or (result.error or "declined")
            records[step].append(produced)
            failures[step].append(broke)
        if index % 10 == 0 or index == len(rows):
            print(f"    {index}/{len(rows)}")

    scores, moves = {}, {}
    for step in range(1, budget + 1):
        score = scoring_repair.RepairScore(arm=f"{name}@{step}" if budget > 1 else name)
        transitions = scoring_repair.Transitions()
        after = scoring.per_document(args.corpus, records[step])
        for position, row in enumerate(rows):
            graded = after.get(row["file"]) or {}
            merged = records[step][position]
            signals = feature_mod.extract(merged, row["spec"], row["variant"],
                                          validators)
            gates_after = len(policy.decide(signals).reasons)
            accuracy = graded.get("field_accuracy")
            ungradable = accuracy is None
            score.add(scoring_repair.Outcome(
                file=row["file"],
                before=row["before"],
                # A failed attempt leaves the document where it was. Scoring a crashed
                # call as a ruined extraction would make an outage read as damage.
                after=row["before"] if ungradable else accuracy,
                gates_before=row["gates_before"], gates_after=gates_after,
                attempts=step, doc_type=row["doc_type"], profile=row["profile"],
                error=failures[step][position] or ("not gradable" if ungradable else ""),
                correct_before=row["correct_before"],
                correct_after=(row["correct_before"] if ungradable
                               else graded.get("fields_correct")),
                fields=row["fields"], layout=row.get("layout")))

            # Field by field, what repair actually did. The document-level delta
            # cannot distinguish "corrected a total" from "invented an address", and
            # a repair can do both at once and net out positive.
            truth = row.get("truth")
            if truth is not None:
                transitions.add(
                    scoring.field_states(row["record"], truth, row["spec"],
                                         row["variant"]),
                    scoring.field_states(merged, truth, row["spec"], row["variant"]),
                    scoring.field_weights(row["spec"], row["variant"]))
        scores[step] = score
        moves[step] = transitions
    return scores, moves


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="repair", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run repair arms and score them")
    run.add_argument("--predictions", required=True)
    run.add_argument("--corpus", default=CORPUS_ROOT)
    run.add_argument("--config", default=None)
    run.add_argument("--only", default="")
    run.add_argument("--arms", default="rerun,reprompt",
                     help="comma-separated. Keep `rerun` in: without it no gain can "
                          "be separated from what a second sample is worth")
    run.add_argument("--extractor", default="")
    run.add_argument("--normalizer", default="")
    run.add_argument("--budget", type=int, default=1,
                     help="model calls per document per arm. Above 1 the arms are "
                          "scored after every attempt, giving a curve rather than two "
                          "endpoints. Every arm gets the same budget, or the "
                          "comparison prices extra sampling instead of the guidance")
    run.add_argument("--limit", type=int, default=0,
                     help="documents per arm. Repair is a model call each, so this is "
                          "required in practice")
    run.add_argument("--all", action="store_false", dest="flagged_only",
                     help="repair every document, not only the ones the router "
                          "flagged. Useful for measuring what repair does to answers "
                          "nothing complained about")
    run.add_argument("--no-validators", action="store_true", dest="no_validators",
                     help="skip re-running the rules. Repair needs them -- the "
                          "complaints are most of what the guided arm is given -- so "
                          "this is for timing the rest of the pipeline, not for a "
                          "measurement")
    run.add_argument("--format", default="table", choices=["table", "json"])
    run.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    config = config_mod.load(args.config)

    rows, validators, policy = _candidates(args, config)
    if not rows:
        raise SystemExit("no documents to repair (nothing flagged, or nothing graded)")
    print(f"  {len(rows)} documents to repair"
          f"{'' if args.flagged_only else ' (every document, not only flagged)'}")

    names = [n.strip() for n in args.arms.split(",") if n.strip()]
    # Budgets are equal by construction now: --budget applies to every arm and each
    # repairer is built with max_attempts=1 so the runner owns the loop. The manifest's
    # own max_attempts is deliberately ignored here, because an arm configured with a
    # larger budget than the one it baselines would price extra sampling as guidance.
    if "rerun" not in names:
        print("  WARNING: no `rerun` arm. Any gain reported below includes whatever a "
              "second sample is worth and cannot be attributed to the feedback.",
              file=sys.stderr)

    budget = max(1, args.budget)
    series, movements = {}, {}
    for name in names:
        series[name], movements[name] = run_arm(name, rows, config, args, validators,
                                                policy, budget)

    # Flatten to arms the scorer understands. At budget 1 the names are unchanged, so
    # nothing about the single-attempt report moves.
    arms = {}
    for name, steps in series.items():
        for step, score in steps.items():
            arms[score.arm] = score
    # The control is free -- it is the extraction that already happened -- so it is
    # always present rather than something a caller can forget. Without it the report
    # can compare arms to each other and cannot say whether any of them should run.
    if arms:
        arms["no_repair"] = scoring_repair.no_repair_arm(
            next(iter(arms.values())).outcomes)
    from core import stamp as stamp_mod

    data = scoring_repair.compare(arms)
    # Stamped before anything is rendered, so a result on disk always says what made
    # it. Two bugs in this phase invalidated whole sets of numbers, and both times the
    # expensive part was working out which artifacts had inherited them.
    data["stamp"] = stamp_mod.stamp(
        args.corpus,
        {"arms": names, "budget": budget, "flagged_only": args.flagged_only},
        files=[r["file"] for r in rows],
        config=config, reader=args.normalizer or None,
        doctypes=sorted({(r["spec"], r["variant"]) for r in rows},
                        key=lambda p: (p[0].name, p[1])))
    print(f"  {stamp_mod.describe(data['stamp'])}")
    if budget > 1:
        data["budget_curve"] = scoring_repair.budget_curve(arms, names, budget)
    # Transitions at the full budget: what the loop did to the fields, end to end.
    data["transitions"] = {name: steps[budget].to_dict()
                           for name, steps in movements.items() if budget in steps}
    slices = {}
    for name, score in arms.items():
        if len(arms) == 1:
            slices["doc_type"] = scoring_repair.by_slice(score, "doc_type")
            slices["profile"] = scoring_repair.by_slice(score, "profile")

    if args.format == "json":
        sys.stdout.write(json.dumps(data, indent=1))
    else:
        print(scoring_repair.render(data, slices))
        if data.get("budget_curve"):
            print(scoring_repair.render_budget_curve(data["budget_curve"]))
        for name, table in (data.get("transitions") or {}).items():
            print(scoring_repair.render_transitions(table, name))

    out = args.out or os.path.join(REPORTS_DIR, "repair.json")
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({**data, "slices": slices,
                       "documents_selected": len(rows),
                       "flagged_only": args.flagged_only}, handle, indent=1)
        if args.format != "json":
            print(f"repair written to {out}")
    except OSError as error:
        print(f"could not write {out}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
