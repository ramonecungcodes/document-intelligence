#!/usr/bin/env python3
"""Run the Phase 1 extractor over a corpus and write predictions.

    python -m extract.cli run --only invoices --limit 20
    python -m extract.cli run --model qwen/qwen3.5-9b --limit 5
    python -m extract.cli run --backend anthropic --model claude-opus-5 --limit 5
    python -m extract.cli schema --type multi_bill_invoice     # no API call

Predictions come out in the same shape as the corpus labels, so scoring them is:

    python -m eval.cli score --predictions /reports/predictions.jsonl

`schema` prints the generated JSON Schema and the system prompt for a type without
calling anything -- useful for seeing what the model is actually being asked for.

The backend is configured from the environment (DI_BACKEND, DI_BASE_URL, DI_MODEL,
DI_API_KEY); see .env.example. `--backend` and `--model` override it for one run,
which is how the same corpus gets scored against two models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from core import doctypes
from eval.score import load_corpus
from extract import schema as schema_mod
from extract import backends
from extract.backends import Usage
from extract.runner import extract_document

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
    if args.dry_run:
        print(f"{len(jobs)} documents (dry run, nothing is called)")
        for doctype, rel in jobs[:10]:
            print(f"  {doctype.name:22} {rel}")
        print(f"  ... {len(jobs)} total")
        return 0

    overrides = {"DI_BACKEND": args.backend, "DI_MODEL": args.model}
    if args.abort_after:
        # Cap the HTTP call too, so a stalled request cannot outlive the budget it is
        # being measured against.
        overrides["DI_TIMEOUT"] = str(args.abort_after)
    if args.no_think:
        overrides["DI_NO_THINK"] = "1"
    backend = backends.from_env(overrides)
    budget = f" · abort past {args.abort_after}s/doc" if args.abort_after else ""
    print(f"{len(jobs)} documents · {backend.describe()} · "
          f"{args.concurrency} at a time{budget}")
    total = Usage()
    done = failed = skipped = 0
    results = []

    def work(job):
        doctype, rel = job
        return extract_document(backend, doctype, os.path.join(corpus_root, rel), rel)

    aborted = ""
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {pool.submit(work, job) for job in jobs}
        try:
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    result = future.result()
                    done += 1
                    total.add(result.usage)
                    results.append(result)
                    name = result.record["file"]
                    if result.error:
                        failed += 1
                        print(f"  [{done}/{len(jobs)}] FAILED {name}: {result.error}",
                              file=sys.stderr, flush=True)
                    elif result.skipped:
                        skipped += 1
                    else:
                        print(f"  [{done}/{len(jobs)}] {name}  {result.usage.seconds:.0f}s",
                              flush=True)

                    if args.abort_after and result.usage.seconds > args.abort_after:
                        aborted = (f"{name} took {result.usage.seconds:.0f}s, over the "
                                   f"{args.abort_after}s budget")
                if aborted:
                    for future in pending:
                        future.cancel()
                    pending = set()
        finally:
            pass

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result.record) + "\n")

    run_path = os.path.splitext(out_path)[0] + ".run.json"
    with open(run_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({
            "backend": backend.name,
            "model": backend.model,
            "no_think": getattr(backend, "no_think", False),
            "endpoint": getattr(backend, "base_url", None),
            "corpus": corpus_root,
            "documents": len(jobs),
            "aborted": aborted or None,
            "failed": failed,
            "skipped_no_text_layer": skipped,
            "usage": total.to_dict(),
        }, handle, indent=2)
        handle.write("\n")

    print()
    if aborted:
        print(f"ABORTED: {aborted}")
        print("  nothing is wrong with the run; the model is just too slow to iterate on.")
        print()
    print(f"wrote {out_path}")
    print(f"      {run_path}")
    print(f"  extracted {done - failed - skipped}/{len(jobs)}"
          + (f", {failed} failed" if failed else "")
          + (f", {skipped} had no text layer" if skipped else ""))
    cost = f"${total.usd:.2f}" if total.usd else "$0.00 (local)"
    reasoning = f" ({total.reasoning_tokens:,} reasoning)" if total.reasoning_tokens else ""
    print(f"  {total.input_tokens:,} in / {total.output_tokens:,} out{reasoning}"
          f"  ·  {cost}  ·  {total.seconds:.0f}s of model time")
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
    go.add_argument("--backend", default="", choices=["", "openai", "anthropic"],
                    help="overrides DI_BACKEND")
    go.add_argument("--model", default="", help="overrides DI_MODEL")
    go.add_argument("--concurrency", type=int, default=4)
    go.add_argument("--abort-after", type=int, default=0, metavar="SECONDS",
                    help="stop the run if any single document takes longer than this")
    go.add_argument("--no-think", action="store_true",
                    help="disable chain-of-thought (overrides DI_NO_THINK)")
    go.add_argument("--dry-run", action="store_true", help="list the work, call nothing")

    show = sub.add_parser("schema", help="print the schema and prompt for a type")
    show.add_argument("--type", required=True)

    args = parser.parse_args(argv)
    args.only = [s.strip() for s in args.only.split(",") if s.strip()] or None \
        if hasattr(args, "only") else None
    return run(args) if args.command == "run" else show_schema(args)


if __name__ == "__main__":
    sys.exit(main())
