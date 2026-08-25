#!/usr/bin/env python3
"""Run the Phase 1 extractor over a corpus and write predictions.

    python -m extract.cli run --only invoices --limit 20
    python -m extract.cli run --corpus /data/degraded --out /reports/degraded.jsonl
    python -m extract.cli schema --type multi_bill_invoice     # no API call

Predictions come out in the same shape as the corpus labels, so scoring them is:

    python -m eval.cli score --predictions /reports/predictions.jsonl

`schema` prints the generated JSON Schema and the system prompt for a type without
calling the API -- useful for seeing what the model is actually being asked for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from core import doctypes
from eval.score import load_corpus
from extract import schema as schema_mod
from extract.runner import MODEL, Result, Usage, build_client, extract_document

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "/data")
REPORTS_DIR = os.environ.get("DI_REPORTS_DIR", "/reports")


def collect(corpus_root, only, limit):
    """Every corpus document, paired with the type declaration it should be read as."""
    jobs, unknown = [], set()
    for stem, records in load_corpus(corpus_root, only).items():
        doctype = doctypes.for_label_file(stem)
        if doctype is None:
            unknown.add(stem)
            continue
        for record in (records[:limit] if limit else records):
            jobs.append((doctype, record["file"]))
    return jobs, unknown


def run(args):
    corpus_root = args.corpus
    jobs, unknown = collect(corpus_root, args.only, args.limit)
    for stem in sorted(unknown):
        print(f"skipping labels/{stem}.json: no document type registered", file=sys.stderr)
    if not jobs:
        raise SystemExit("nothing to extract")

    out_path = args.out or os.path.join(REPORTS_DIR, "predictions.jsonl")
    print(f"{len(jobs)} documents · model {MODEL} · effort {args.effort} "
          f"· {args.concurrency} at a time")
    if args.dry_run:
        for doctype, rel in jobs[:10]:
            print(f"  {doctype.name:22} {rel}")
        print(f"  ... {len(jobs)} total (dry run, no API calls)")
        return 0

    client = build_client()
    total = Usage()
    done = failed = skipped = 0
    results = []

    def work(job):
        doctype, rel = job
        return extract_document(client, doctype, os.path.join(corpus_root, rel), rel,
                                effort=args.effort)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for result in pool.map(work, jobs):
            done += 1
            total.add(result.usage)
            results.append(result)
            if result.error:
                failed += 1
                print(f"  [{done}/{len(jobs)}] FAILED {result.record['file']}: {result.error}",
                      file=sys.stderr)
            elif result.skipped:
                skipped += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] ${total.usd:.2f} spent", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result.record) + "\n")

    run_path = os.path.splitext(out_path)[0] + ".run.json"
    with open(run_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({
            "model": MODEL,
            "effort": args.effort,
            "corpus": corpus_root,
            "documents": len(jobs),
            "failed": failed,
            "skipped_no_text_layer": skipped,
            "usage": total.to_dict(),
        }, handle, indent=2)
        handle.write("\n")

    print()
    print(f"wrote {out_path}")
    print(f"      {run_path}")
    print(f"  extracted {done - failed - skipped}/{len(jobs)}"
          + (f", {failed} failed" if failed else "")
          + (f", {skipped} had no text layer" if skipped else ""))
    print(f"  {total.input_tokens:,} in / {total.output_tokens:,} out"
          f"  ·  ${total.usd:.2f}  ·  {total.seconds:.0f}s of model time")
    print()
    print(f"score it:  python -m eval.cli score --predictions {out_path}")
    return 0


def show_schema(args):
    doctype = doctypes.REGISTRY.get(args.type)
    if doctype is None:
        raise SystemExit(f"unknown type {args.type!r}; known: {', '.join(sorted(doctypes.REGISTRY))}")
    print(schema_mod.instructions(doctype))
    print()
    print(json.dumps(schema_mod.json_schema(doctype), indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="extract", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    go = sub.add_parser("run", help="extract every document in the corpus")
    go.add_argument("--corpus", default=CORPUS_ROOT)
    go.add_argument("--only", default="", help="comma-separated label stems")
    go.add_argument("--limit", type=int, default=0, help="first N documents per type")
    go.add_argument("--out", default=None, help=f"default: {REPORTS_DIR}/predictions.jsonl")
    go.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    go.add_argument("--concurrency", type=int, default=4)
    go.add_argument("--dry-run", action="store_true", help="list the work, call nothing")

    show = sub.add_parser("schema", help="print the schema and prompt for a type")
    show.add_argument("--type", required=True)

    args = parser.parse_args(argv)
    args.only = [s.strip() for s in args.only.split(",") if s.strip()] or None \
        if hasattr(args, "only") else None
    return run(args) if args.command == "run" else show_schema(args)


if __name__ == "__main__":
    sys.exit(main())
