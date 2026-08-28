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
    """The documents to repair, and their complaints -- chosen once for every arm."""
    import repair
    from route import features as feature_mod
    from route import policy as policy_mod
    from validate.base import build_all

    records = scoring.load_records(args.predictions)
    only = [t.strip() for t in args.only.split(",") if t.strip()] or None
    graded = scoring.per_document(args.corpus, records, only)
    validators = build_all(config)
    policy = policy_mod.build(config)

    truth_of = {}
    for stem, rows in scoring.load_corpus(args.corpus, only).items():
        spec = doctypes.for_label_file(stem)
        for row in rows:
            key = str(row.get("file", "")).replace("\\", "/")
            if key:
                truth_of[key] = (stem, spec, spec.variant_of(row) if spec else "")

    out = []
    for record in records:
        key = str(record.get("file", "")).replace("\\", "/")
        before = graded.get(key)
        if key not in truth_of or before is None or before["failed"]:
            continue
        if before["field_accuracy"] is None:
            continue
        stem, spec, variant = truth_of[key]
        signals = feature_mod.extract(record, spec, variant, validators)
        decision = policy.decide(signals)
        if args.flagged_only and not decision.review:
            continue
        out.append({
            "file": key, "doc_type": stem, "spec": spec, "variant": variant,
            "profile": feature_mod.profile_of(key),
            "record": record,
            "before": before["field_accuracy"],
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


def run_arm(name, rows, config, args, validators, policy):
    """One arm over every candidate. Returns a RepairScore."""
    import repair
    from route import features as feature_mod
    from extract import backends

    repairer = repair.build(name, config)
    print(f"  arm {name}: {repairer.describe()}")
    backend = backends.build(config=config, plugin=args.extractor)
    normalizer = _normalizer_for(config, args.normalizer, args.corpus)
    print(f"  normalizer: {normalizer.describe()}")

    # Repair must re-read the page the way the original extraction did. Reading it with
    # a better engine would show up as a repair gain that a prompt had nothing to do
    # with -- the single most attributable-looking wrong number this stage can produce.
    # The engines the predictions recorded are known, so the mismatch is checkable.
    was = {str((row["record"].get("_normalizer") or {}).get("engine") or "?")
           for row in rows}
    now = normalizer.describe().split()[0]
    if was and not any(now.split(":")[0] in engine for engine in was):
        print(f"  WARNING: these predictions were read with {sorted(was)} and repair "
              f"is reading with {now!r}. Any gain below may be the reader, not the "
              f"repair.", file=sys.stderr)

    score = scoring_repair.RepairScore(arm=name)
    repaired_records = []
    for index, row in enumerate(rows, 1):
        path = os.path.join(args.corpus, row["file"])
        page = normalizer.read(path)
        context = repair.Context(
            backend=backend, doctype=row["spec"], variant=row["variant"],
            path=path, relative_path=row["file"], record=row["record"],
            text=page.text, complaints=row["complaints"], normalizer=normalizer)
        result = repairer.repair(context)

        if result.changed:
            # The repaired record keeps the file key and the harness provenance; only
            # the model's own fields are replaced. A repair that dropped `file` would
            # silently fail to join against the corpus and score as a missing document.
            merged = {k: v for k, v in row["record"].items()
                      if k.startswith("_") or k == "file"}
            merged["doc_type"] = row["spec"].name
            merged.update(result.record)
            if row["variant"] and row["spec"].variant_key:
                merged[row["spec"].variant_key] = row["variant"]
        else:
            merged = dict(row["record"])
        repaired_records.append(merged)

        if index % 10 == 0 or index == len(rows):
            print(f"    {index}/{len(rows)}")

    after = scoring.per_document(args.corpus, repaired_records)
    for row, merged in zip(rows, repaired_records):
        graded = after.get(row["file"]) or {}
        signals = feature_mod.extract(merged, row["spec"], row["variant"], validators)
        gates_after = len(policy.decide(signals).reasons)
        accuracy = graded.get("field_accuracy")
        score.add(scoring_repair.Outcome(
            file=row["file"],
            before=row["before"],
            # A repair that failed leaves the document exactly where it was. Scoring a
            # crashed call as a ruined extraction would make an outage read as a
            # damaging loop.
            after=row["before"] if accuracy is None else accuracy,
            gates_before=row["gates_before"], gates_after=gates_after,
            attempts=1, doc_type=row["doc_type"], profile=row["profile"]))
    return score


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
    run.add_argument("--limit", type=int, default=0,
                     help="documents per arm. Repair is a model call each, so this is "
                          "required in practice")
    run.add_argument("--all", action="store_false", dest="flagged_only",
                     help="repair every document, not only the ones the router "
                          "flagged. Useful for measuring what repair does to answers "
                          "nothing complained about")
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
    if "rerun" not in names:
        print("  WARNING: no `rerun` arm. Any gain reported below includes whatever a "
              "second sample is worth and cannot be attributed to the feedback.",
              file=sys.stderr)

    arms = {name: run_arm(name, rows, config, args, validators, policy)
            for name in names}
    data = scoring_repair.compare(arms)
    slices = {}
    for name, score in arms.items():
        if len(arms) == 1:
            slices["doc_type"] = scoring_repair.by_slice(score, "doc_type")
            slices["profile"] = scoring_repair.by_slice(score, "profile")

    if args.format == "json":
        sys.stdout.write(json.dumps(data, indent=1))
    else:
        print(scoring_repair.render(data, slices))

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
