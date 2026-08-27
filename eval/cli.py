#!/usr/bin/env python3
"""Score extractions against the corpus.

    score      grade predictions and print a table (and write report.json)
    selftest   score the ground truth against itself; must come out at 1.000

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
import os
import sys

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

    args = parser.parse_args(argv)
    only = [s.strip() for s in args.only.split(",") if s.strip()] or None

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
