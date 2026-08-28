#!/usr/bin/env python3
"""Score extractions against the corpus.

    score      grade predictions and print a table (and write report.json)
    selftest   score the ground truth against itself; must come out at 1.000
    calibrate  ask whether the confidence is real, and where the floor belongs

`calibrate --against extraction` is the one worth quoting: it asks whether confidence
predicts the fields coming back right, which is what a floor is really deciding.
`--against classification` asks only whether the type was named correctly, and a
pipeline can be excellent at that while the documents it types confidently still
extract badly.

Usage:
    python -m eval.cli score --predictions preds.jsonl
    python -m eval.cli score --predictions preds.jsonl --only invoices --format json
    python -m eval.cli score --predictions empty         # the null-extractor baseline
    python -m eval.cli selftest

`--predictions none` grades an extractor that returns nothing, which should score
0.000. `selftest` feeds the labels back in as predictions, which must score exactly
1.000 -- if it does not, a normaliser is wrong and every later number is quietly
deflated by a bug nobody would go looking for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from eval import calibration
from eval import score as scoring
from eval.report import ScoreReport

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "/data")
REPORTS_DIR = os.environ.get("DI_REPORTS_DIR", "/reports")


def _fmt(value, width=7):
    return f"{value:>{width}.3f}" if isinstance(value, float) else f"{'--':>{width}}"


def _field_note(f) -> str:
    """What is worth saying about a field beyond its accuracy.

    Invented values are never elided. This used to be an elif chain in the top-level
    table that could only ever say "missing", and no note at all in the group table --
    so a field inventing a co-applicant name on all 25 loan applications that had none
    rendered as a bare 0.375 with an empty column, indistinguishable from a field that
    simply missed things, and the opposite diagnosis. The counter was there the whole
    time; only the rendering hid it.

    Shared by both tables. Having two copies of this logic is what let them drift.
    """
    notes = []
    if f["recovered_by_normalisation"]:
        kinds = ", ".join(f["notes"]) or "normalised"
        notes.append(f"+{f['recovered_by_normalisation']} by {kinds}")
    if f["spurious"]:
        notes.append(f"{f['spurious']} invented")
    if f["missing"]:
        notes.append(f"{f['missing']} missing")
    return ", ".join(notes)


def _walk_fields(container, prefix=""):
    """Every field in a slice, including those nested inside repeating groups."""
    for f in container.get("fields", []):
        yield prefix + f["field"], f
    for group in container.get("groups", []):
        yield from _walk_fields(group, prefix + group["group"] + ".")


def _render_abstention(out, data) -> None:
    """Fields the document is allowed to omit, scored on whether we noticed.

    Kept apart from the accuracy table because it answers a different question. An
    accuracy figure charges the same one point for a missed value and an invented one,
    which makes the worse error the invisible one: a blank field is honest and gets
    looked at, while a confident wrong one flows downstream unchallenged. Only fields
    that are genuinely absent somewhere in the corpus appear here -- for the rest there
    is no abstention decision to get wrong.
    """
    rows = []
    for slice_row in data.get("slices", []):
        if slice_row.get("dimension") != "doc_type":
            continue
        for name, f in _walk_fields(slice_row):
            if f.get("n_absent"):
                rows.append((slice_row["slice"], name, f))
    if not rows:
        return
    out.append("")
    out.append("ABSTENTION  ·  fields the document may legitimately omit")
    out.append(f"  {'field':<34}{'absent':>7}{'present':>8}{'presence':>10}"
               f"{'precision':>11}{'invented':>10}{'copied':>8}")
    for slice_name, name, f in sorted(rows, key=lambda r: (r[2].get("false_positive_rate") or 0),
                                      reverse=True):
        out.append(
            f"  {name[:33]:<34}{f['n_absent']:>7}{f['n_present']:>8}"
            f"{_fmt(f.get('presence_accuracy'), 10)}"
            f"{_fmt(f.get('precision_populated'), 11)}"
            f"{_fmt(f.get('false_positive_rate'), 10)}"
            f"{_fmt(f.get('contamination_rate'), 8)}"
        )
    out.append("  invented = a value where the document had none; copied = that value")
    out.append("  taken verbatim from a neighbouring field.")


def render(report: ScoreReport) -> str:
    data = report.to_dict()
    out = []
    overall = data["overall"]

    out.append("")
    out.append(f"{'OVERALL':<24}{'':>10}")
    out.append(f"  documents          {overall['documents']:>8}")
    out.append(f"  scored             {overall['scored']:>8}")
    if overall.get("failed"):
        out.append(f"  extraction failed  {overall['failed']:>8}   (not graded)")
    out.append(f"  fields graded      {overall['fields_graded']:>8}")
    out.append(f"  field accuracy     {_fmt(overall['field_accuracy'], 8)}")
    out.append(f"  ... excluding blank{_fmt(overall['field_accuracy_nonblank'], 8)}"
               f"   ({overall['blank_fields']} fields blank in truth)")
    out.append(f"  exact match        {_fmt(overall['field_exact'], 8)}")

    for dimension, heading in (
        ("doc_type", "BY DOCUMENT TYPE"),
        ("degradation", "BY DEGRADATION"),
        ("layout", "BY LAYOUT"),
    ):
        rows = [s for s in data["slices"] if s["dimension"] == dimension]
        if not rows:
            continue
        out.append("")
        out.append(heading)
        out.append(f"  {'slice':<24}{'docs':>6}{'ok':>5}{'fail':>6}"
                   f"{'accuracy':>10}{'non-blank':>11}{'exact':>9}")
        for row in rows:
            out.append(
                f"  {row['slice']:<24}{row['documents']:>6}"
                f"{row['scored']:>5}{row.get('failed', 0):>6}"
                f"{_fmt(row['field_accuracy'], 10)}"
                f"{_fmt(row['field_accuracy_nonblank'], 11)}"
                f"{_fmt(row['field_exact'], 9)}"
            )

    for row in [s for s in data["slices"] if s["dimension"] == "doc_type"]:
        if not row["fields"] and not row["groups"]:
            continue
        out.append("")
        out.append(f"{row['slice'].upper()}  ·  {row['documents']} docs")
        out.append(f"  {'field':<26}{'n':>6}{'exact':>8}{'accuracy':>10}  note")
        for f in sorted(row["fields"], key=lambda x: x["field"]):
            note = _field_note(f)
            out.append(
                f"  {f['field']:<26}{f['n']:>6}{_fmt(f['exact'], 8)}"
                f"{_fmt(f['accuracy'], 10)}  {note}"
            )
        for group in row["groups"]:
            _render_group(out, group, indent=2)

    _render_abstention(out, data)

    detection = data.get("detection")
    if detection:
        out.append("")
        out.append("DEFECT DETECTION")
        out.append(f"  precision          {_fmt(detection['precision'], 8)}")
        out.append(f"  recall             {_fmt(detection['recall'], 8)}")
        out.append(
            f"  false positives on clean  {_fmt(detection['false_positive_rate_on_clean'], 8)}"
            f"   ({detection['clean_documents_flagged']}/{detection['clean_documents']} docs)"
        )

    if data["warnings"]:
        out.append("")
        out.append("WARNINGS")
        for warning in data["warnings"]:
            out.append(f"  - {warning}")

    out.append("")
    return "\n".join(out)


def _render_group(out, group, indent=2):
    pad = " " * indent
    out.append("")
    out.append(
        f"{pad}{group['group']}  ·  {group['truth_rows']} rows"
        f"   recall {_fmt(group['row_recall'], 6)}"
        f"   precision {_fmt(group['row_precision'], 6)}"
        f"   f1 {_fmt(group['row_f1'], 6)}"
    )
    for f in sorted(group["fields"], key=lambda x: x["field"]):
        out.append(
            f"{pad}  {f['field']:<24}{f['n']:>6}{_fmt(f['exact'], 8)}"
            f"{_fmt(f['accuracy'], 10)}  {_field_note(f)}".rstrip()
        )
    for nested in group["groups"]:
        _render_group(out, nested, indent + 2)


def build_predictions(arg: str, corpus_root: str, only):
    """Three synthetic sources, plus a real file.

    `none`  supplies no predictions at all -- the corpus is left ungraded.
    `empty` supplies one empty record per document: an extractor that ran and found
            nothing. This is the baseline worth quoting, and it is not the same thing.
    `self`  feeds the ground truth back in; must score 1.000.
    """
    if arg == "none":
        return []
    if arg == "empty":
        corpus = scoring.load_corpus(corpus_root, only)
        return [{"file": r["file"]} for records in corpus.values() for r in records]
    if arg == "self":
        corpus = scoring.load_corpus(corpus_root, only)
        return [record for records in corpus.values() for record in records]
    if not os.path.exists(arg):
        raise SystemExit(f"predictions file not found: {arg}")
    return scoring.load_records(arg)


def _profile_of(relative_path: str) -> str:
    """Which degradation this document is, read off the path.

    Derived rather than stored, on purpose. The corpus names degraded documents
    `<stem>__<profile>.pdf` and puts them under `degraded/`, so the filename already
    carries this; writing it into the signals file as well would be a second copy of
    one fact, and the docTR cache is the standing lesson on where that ends.
    """
    stem = os.path.basename(relative_path)
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    _, sep, profile = stem.partition("__")
    return profile if sep else "clean"


def _truth_labels(corpus_root: str, only=None) -> dict:
    """file -> `type:variant`, the label a classifier is graded against.

    Type and variant together, because the variant is what selects the field set. A
    curve drawn on the type alone would count a W-9 read as an onboarding form a
    correct answer, and then report the high confidence on it as well calibrated.
    """
    from core import doctypes

    labels = {}
    for stem, records in scoring.load_corpus(corpus_root, only).items():
        doctype = doctypes.for_label_file(stem)
        for record in records:
            key = str(record.get("file", "")).replace("\\", "/")
            if not key:
                continue
            base = record.get("doc_type") or (doctype.name if doctype else "")
            variant = doctype.variant_of(record) if doctype else ""
            labels[key] = f"{base}:{variant}" if variant else base
    return labels


def extraction_observations(args, score):
    """Confidence against how well the document actually extracted.

    This is the join Phase 5 exists for. Whether the classifier named the right type is
    not what a floor decides -- a floor decides whether a person has to look at the
    document, and that turns on whether its *fields* came back right. The two questions
    can disagree in both directions: a correctly typed fax whose text is unreadable
    extracts badly, and a misfiled purchase order can still yield most of its fields
    because an invoice schema and a PO schema overlap heavily.

    The outcome is the document's field accuracy, not a pass/fail. Choosing a bar here
    would put a number nobody could see in front of every figure downstream.

    A failed extraction is dropped rather than scored zero, and that is deliberate: a
    crash and an extractor that ran and got everything wrong are different events, and
    calling one the other would let an outage read as a model regression. They are
    counted and reported instead.
    """
    from route import signals as signals_mod

    rows = signals_mod.read(args.signals)
    if not rows:
        raise SystemExit(
            f"no signals beside {args.signals}.\n"
            f"  Expected {signals_mod.path_for(args.signals)}, which "
            f"`extract.cli run --type-from classifier` writes.")
    if not os.path.exists(args.signals):
        raise SystemExit(f"predictions file not found: {args.signals}")
    only = [t.strip() for t in args.only.split(",") if t.strip()] or None
    graded = scoring.per_document(args.corpus, scoring.load_records(args.signals), only)

    failed = ungraded = 0
    for key, row in sorted(rows.items()):
        outcome = graded.get(key)
        if outcome is None:
            ungraded += 1
            continue
        if outcome["failed"] or outcome["field_accuracy"] is None:
            failed += 1
            continue
        classifier = row.get("classifier") or {}
        answer = classifier.get("withheld") or classifier.get("doc_type") or ""
        score.add(classifier.get("confidence"), outcome["field_accuracy"],
                  _profile_of(key), truth=outcome["doc_type"], answer=answer)
    if failed:
        print(f"  {failed} extractions failed and are not scored; a crash is not a "
              f"wrong answer", file=sys.stderr)
    if ungraded:
        print(f"  {ungraded} documents in the signals file have no graded extraction; "
              f"skipped", file=sys.stderr)
    return score


def observations(args):
    """Build the decision set from whichever artifact was named.

    Two sources, one shape. A classifier report is what the training tool writes, and
    is the only place a *design* holdout exists -- the split that showed the image
    model memorising templates -- so calibration measured anywhere else is measured on
    documents the model has effectively seen. A signals sidecar is the pipeline's own
    record, on whatever corpus was actually run. They answer different questions and
    both are worth asking, which is why neither is the default.
    """
    against = getattr(args, "against", "classification")
    score = calibration.CalibrationScore(outcome_of=against,
                                         error_below=args.error_below)
    if against == "extraction":
        return extraction_observations(args, score)
    if args.report:
        with open(args.report, encoding="utf-8") as handle:
            report = json.load(handle)
        if isinstance(report, list):
            raise SystemExit(f"{args.report} is not a classifier report; expected an "
                             f"object with a 'documents' list")
        for row in report.get("documents", []):
            # `predicted` here is what the model said, before any floor. The training
            # tool writes the raw answer, which is exactly what a coverage curve needs
            # and what a floored record cannot supply.
            score.add(row.get("confidence"),
                      row.get("predicted") == row.get("truth"),
                      row.get("profile") or _profile_of(row.get("file", "")),
                      truth=row.get("truth", ""), answer=row.get("predicted", ""))
        return score

    from route import signals as signals_mod

    rows = signals_mod.read(args.signals)
    if not rows:
        raise SystemExit(
            f"no signals beside {args.signals}.\n"
            f"  Expected {signals_mod.path_for(args.signals)}, which "
            f"`extract.cli run --type-from classifier` writes.\n"
            f"  Runs made before this stage existed have none; re-run to get them.")
    only = [t.strip() for t in args.only.split(",") if t.strip()] or None
    truth = _truth_labels(args.corpus, only)
    missing = 0
    for key, row in sorted(rows.items()):
        classifier = row.get("classifier") or {}
        if key not in truth:
            missing += 1
            continue
        # The answer, whether or not it survived the floor. `withheld` is what the
        # classifier was going to say before abstention blanked it, and skipping those
        # documents would grade the floor against the ones it already let through --
        # which reports every floor as costless.
        answer = classifier.get("withheld") or ""
        if not answer and classifier.get("doc_type"):
            answer = classifier["doc_type"]
            if classifier.get("variant"):
                answer += ":" + classifier["variant"]
        score.add(classifier.get("confidence"), answer == truth[key],
                  _profile_of(key), truth=truth[key], answer=answer)
    if missing:
        print(f"  {missing} documents in the signals file have no label in the "
              f"corpus; skipped", file=sys.stderr)
    return score


def main(argv=None):
    parser = argparse.ArgumentParser(prog="eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("score", help="grade predictions against the corpus")
    run.add_argument("--predictions", required=True,
                     help="path to predictions (JSON array or JSONL), or 'none' / 'empty' / 'self'")
    run.add_argument("--corpus", default=CORPUS_ROOT, help="corpus root (default: %(default)s)")
    run.add_argument("--only", default="", help="comma-separated label stems, e.g. invoices,forms")
    run.add_argument("--format", default="table", choices=["table", "json"])
    run.add_argument("--out", default=None,
                     help=f"where to write report.json (default: {REPORTS_DIR}/report.json)")
    run.add_argument("--label", default=None, help="a name for this run, recorded in provenance")

    check = sub.add_parser("selftest", help="score ground truth against itself; must be 1.000")
    check.add_argument("--corpus", default=CORPUS_ROOT)
    check.add_argument("--only", default="")

    cal = sub.add_parser("calibrate",
                         help="is the confidence real, and where does the floor go")
    source = cal.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", default="",
                        help="a classifier report from tools/train-layout-classifier.py "
                             "-- the only artifact carrying a design holdout")
    source.add_argument("--signals", default="",
                        help="a predictions file whose .signals.jsonl sidecar holds "
                             "what the pipeline knew while deciding")
    cal.add_argument("--corpus", default=CORPUS_ROOT)
    cal.add_argument("--only", default="")
    cal.add_argument("--against", default="classification",
                     choices=["classification", "extraction"],
                     help="what the outcome measures: whether the type was right, or "
                          "how much of the document extracted correctly. The second "
                          "needs --signals. Default: %(default)s")
    cal.add_argument("--error-below", type=float, default=1.0, dest="error_below",
                     help="a document counts as an error when its outcome is under "
                          "this. 1.0 means every graded field has to be right "
                          "(default: %(default)s)")
    cal.add_argument("--target", type=float, default=0.99,
                     help="the accuracy a floor has to hold (default: %(default)s)")
    cal.add_argument("--format", default="table", choices=["table", "json"])
    cal.add_argument("--out", default=None, help="where to write calibration.json")

    args = parser.parse_args(argv)
    only = [s.strip() for s in args.only.split(",") if s.strip()] or None

    if args.command == "calibrate":
        if args.against == "extraction" and not args.signals:
            # A classifier report holds no extractions, so this combination has no
            # answer rather than a degraded one.
            parser.error("--against extraction needs --signals: a classifier report "
                         "records what the type was, not what the fields came back as")
        score = observations(args)
        data = score.to_dict(args.target)
        if args.format == "json":
            sys.stdout.write(json.dumps(data, indent=1))
        else:
            print(calibration.render(score, args.target))
        out = args.out or os.path.join(REPORTS_DIR, "calibration.json")
        try:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, indent=1)
            if args.format != "json":
                print(f"calibration written to {out}")
        except OSError as error:
            print(f"could not write {out}: {error}", file=sys.stderr)
        return 0

    if args.command == "selftest":
        report = scoring.score(args.corpus, build_predictions("self", args.corpus, only),
                               only=only, provenance={"run": "selftest"})
        overall = report.overall()
        accuracy = overall["field_accuracy"]
        print(render(report))
        if accuracy is None:
            print("selftest FAILED: nothing was graded")
            return 1
        if accuracy < 1.0:
            print(f"selftest FAILED: ground truth scored {accuracy:.4f}, expected 1.0000")
            print("a normaliser is wrong; every later number would be understated")
            return 1
        print("selftest passed: ground truth scores 1.0000 against itself")
        return 0

    predictions = build_predictions(args.predictions, args.corpus, only)
    report = scoring.score(
        args.corpus, predictions, only=only,
        provenance={
            "run": args.label,
            "corpus_root": args.corpus,
            "predictions": args.predictions,
            "only": only,
            "model": None,
            "plugin_manifest": None,
            "knowledge_pack": None,
        },
    )

    if args.format == "json":
        sys.stdout.write(report.to_json())
    else:
        print(render(report))

    out = args.out or os.path.join(REPORTS_DIR, "report.json")
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        report.write(out)
        if args.format != "json":
            print(f"report written to {out}")
    except OSError as error:
        print(f"could not write {out}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
