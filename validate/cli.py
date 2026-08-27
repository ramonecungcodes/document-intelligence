"""Run the validators and score them, first against truth and then against output.

    python -m validate.cli selftest
    python -m validate.cli run --predictions /reports/preds.jsonl

`selftest` is not a smoke test. It runs every rule over the corpus labels, where the
document is exactly what the label says it is, so any rule that fires on a clean
document has been caught being wrong with no extractor to hide behind. It has to come
out with zero false alarms before the second command's number means anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from core import config as config_mod
from core import doctypes
from eval.score import load_corpus
from eval.validation import ValidationScore, render
from validate.base import VALIDATORS, build_all, run as run_validators

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "/data")
REPORTS_DIR = os.environ.get("DI_REPORTS_DIR", "/reports")


def score_records(validators, records_by_stem, corpus_root):
    score = ValidationScore()
    findings = []
    for stem, records in records_by_stem.items():
        doctype = doctypes.for_label_file(stem)
        if doctype is None:
            continue
        for record in records:
            report = run_validators(validators, record, doctype,
                                    doctype.variant_of(record))
            injected = set(record.get("irregularities") or [])
            errors = {f.code for f in report.findings if f.severity == "error"}
            score.add(report.codes, injected, is_clean=not injected, errors=errors)
            if report.findings:
                findings.append({"file": record.get("file"),
                                 "injected": sorted(injected),
                                 **report.to_dict()})
    return score, findings


def run_predictions(args) -> int:
    """The rules against what the extractor actually produced.

    This is the number a person experiences, and it is the product of two things: the
    rule and the extraction. That is why the self-test comes first -- with rule
    correctness already established at zero false alarms on ground truth, a false alarm
    here has one remaining explanation, and it is the extractor.
    """
    config = config_mod.load(args.config)
    validators = build_all(config)
    corpus_root = args.corpus or CORPUS_ROOT

    truth = {}
    for stem, records in load_corpus(corpus_root, args.only).items():
        doctype = doctypes.for_label_file(stem)
        if doctype is None:
            continue
        for record in records:
            truth[record["file"].replace("\\", "/")] = (doctype, record)

    score = ValidationScore()
    findings, unmatched = [], 0
    with open(args.predictions, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            predicted = json.loads(line)
            key = str(predicted.get("file", "")).replace("\\", "/")
            if key not in truth:
                unmatched += 1
                continue
            doctype, actual = truth[key]
            report = run_validators(validators, predicted, doctype,
                                    doctype.variant_of(actual))
            injected = set(actual.get("irregularities") or [])
            errors = {f.code for f in report.findings if f.severity == "error"}
            score.add(report.codes, injected, is_clean=not injected, errors=errors)
            if report.findings:
                findings.append({"file": key, "injected": sorted(injected),
                                 **report.to_dict()})

    print(f"validators: {', '.join(v.name for v in validators)}")
    print(render(score, against=f"extracted output ({os.path.basename(args.predictions)})"))
    if unmatched:
        print(f"\n  {unmatched} predictions matched no corpus document")
    if args.out:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, os.path.basename(args.out))
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"predictions": args.predictions, "score": score.to_dict(),
                       "findings": findings}, handle, indent=1)
        print(f"\n  wrote {path}")
    return 0


def selftest(args) -> int:
    config = config_mod.load(args.config)
    validators = build_all(config)
    root = args.corpus or CORPUS_ROOT
    print(f"validators: {', '.join(v.name for v in validators)}")

    total = 0
    for label, sub in (("clean corpus", ""), ("defective corpus", "irregular")):
        corpus_root = os.path.join(root, sub) if sub else root
        if not os.path.isdir(os.path.join(corpus_root, "labels")):
            continue
        records = load_corpus(corpus_root, args.only)
        score, findings = score_records(validators, records, corpus_root)
        print(render(score, against=f"labels ({label})"))
        total += score.clean_documents_flagged
        if args.out:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            name = f"{os.path.splitext(os.path.basename(args.out))[0]}-{sub or 'clean'}.json"
            with open(os.path.join(REPORTS_DIR, name), "w",
                      encoding="utf-8", newline="\n") as handle:
                json.dump({"score": score.to_dict(), "findings": findings},
                          handle, indent=1)
            print(f"\n  wrote {os.path.join(REPORTS_DIR, name)}")

    print()
    if total:
        print(f"SELF-TEST FAILED: {total} clean documents were flagged on their own "
              f"ground truth.\n  Those are rules being wrong, not documents being bad.")
        return 1
    print("self-test clean: no rule fires on a document the corpus says is fine.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="validate")
    sub = parser.add_subparsers(dest="command", required=True)

    st = sub.add_parser("selftest", help="run every rule against the corpus labels")
    st.add_argument("--corpus", default="")
    st.add_argument("--only", default="")
    st.add_argument("--out", default="")
    st.add_argument("--config", default="")

    rp = sub.add_parser("run", help="run the rules against extracted output")
    rp.add_argument("--predictions", required=True)
    rp.add_argument("--corpus", default="")
    rp.add_argument("--only", default="")
    rp.add_argument("--out", default="")
    rp.add_argument("--config", default="")

    ls = sub.add_parser("rules", help="show every validator")
    ls.add_argument("--config", default="")

    args = parser.parse_args(argv)
    if hasattr(args, "only"):
        args.only = [p.strip() for p in (args.only or "").split(",") if p.strip()] or None

    if args.command == "selftest":
        return selftest(args)
    if args.command == "run":
        return run_predictions(args)
    if args.command == "rules":
        for name, cls in sorted(VALIDATORS.items()):
            print(f"  {name:<16}{cls.__doc__.splitlines()[0] if cls.__doc__ else ''}")
            if cls.applies_to:
                print(f"                  applies to: {', '.join(cls.applies_to)}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
