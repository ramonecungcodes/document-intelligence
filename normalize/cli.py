"""Run the normalizer over a corpus and cache what it produces.

    python -m normalize.cli run --engine tesseract
    python -m normalize.cli run --engine doctr --only invoices --limit 20
    python -m normalize.cli run --engine cascade --corpus /data/degraded
    python -m normalize.cli engines
    python -m normalize.cli show --engine tesseract invoices/northwind_INV-20261000.pdf

This is a separate stage from extraction on purpose. OCR is expensive and
deterministic; extraction is cheap and the thing being varied. Caching the expensive
half means an ablation across four models pays for OCR once rather than four times.

Runs are resumable. A document already in the cache is skipped unless `--force`, so an
interrupted pass over a thousand documents continues instead of restarting -- which
matters when a pass takes an hour and the machine is also serving a model.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import sys
import time

from core import config as config_mod
from core import doctypes
from eval.score import load_corpus
from normalize import store
from normalize.base import NORMALIZERS, build

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "/data")
NORMALIZED_DIR = os.environ.get("DI_NORMALIZED_DIR", "/data/normalized")


def documents(corpus_root: str, only, limit: int):
    """Every document in the corpus, as corpus-relative paths."""
    found = []
    for stem, records in load_corpus(corpus_root, only).items():
        if doctypes.for_label_file(stem) is None:
            continue
        for record in (records[:limit] if limit else records):
            found.append(record["file"])
    return found


def run(args) -> int:
    config = config_mod.load(args.config)
    engine = build(config=config, plugin=args.engine)
    corpus_root = args.corpus or CORPUS_ROOT
    root = args.out or NORMALIZED_DIR
    name = args.engine or (config.chosen("normalizer") or "native")

    jobs = documents(corpus_root, args.only, args.limit)
    if not args.force:
        pending = [j for j in jobs if not store.exists(root, name, j)]
    else:
        pending = jobs
    skipped = len(jobs) - len(pending)

    print(f"  normalizer: {engine.describe()}")
    print(f"  corpus:     {corpus_root}")
    print(f"  cache:      {os.path.join(root, name)}")
    print(f"  documents:  {len(jobs)}" + (f"  ({skipped} already cached)" if skipped else ""))
    if args.dry_run:
        for job in pending[:20]:
            print(f"    {job}")
        if len(pending) > 20:
            print(f"    ... and {len(pending) - 20} more")
        return 0
    if not pending:
        print("  nothing to do")
        return 0

    started = time.time()
    done = failed = empty = 0
    characters = 0
    confidences = []

    def work(relative):
        result = engine.read(os.path.join(corpus_root, relative))
        store.write(root, name, relative, result)
        return relative, result

    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        running = [pool.submit(work, job) for job in pending]
        for future in futures.as_completed(running):
            try:
                relative, result = future.result()
            except Exception as error:
                failed += 1
                print(f"  [{done + failed}/{len(pending)}] FAILED: "
                      f"{type(error).__name__}: {error}", flush=True)
                continue
            done += 1
            characters += len(result.text)
            if result.confidence is not None:
                confidences.append(result.confidence)
            if result.empty:
                empty += 1
            if done % 25 == 0 or done == len(pending):
                print(f"  [{done}/{len(pending)}] {relative}"
                      f"  {len(result.text)}ch", flush=True)

    elapsed = time.time() - started
    mean = sum(confidences) / len(confidences) if confidences else None
    print()
    print(f"  normalized {done}/{len(pending)}" + (f", {failed} failed" if failed else ""))
    # An empty result is not a failure -- it is the honest answer for a page with
    # nothing on it -- but it is the number that says whether this engine can read this
    # corpus at all, so it is reported rather than folded into the total.
    print(f"  {empty} produced no text at all")
    print(f"  {characters:,} characters" + (f"  ·  mean confidence {mean:.3f}" if mean else ""))
    print(f"  {elapsed:.0f}s  ·  {elapsed / max(done, 1):.1f}s per document")
    print()
    print(f"  extract from it:  DI_NORMALIZER=cached DI_CACHED_ENGINE={name} "
          f"docker compose run --rm extractor run")
    return 0 if not failed else 1


def engines(args) -> int:
    """Every registered normalizer, its settings, and whether this image can run it."""
    from core.plugins import describe
    config = config_mod.load(args.config)
    chosen = config.chosen("normalizer") or "native"
    for name, cls in sorted(NORMALIZERS.items()):
        mark = "  <- selected" if name == chosen else ""
        print(f"--- {name}{mark}")
        print(describe(name, cls.SETTINGS))
        print()
    return 0


def show(args) -> int:
    """Print what an engine produced for one document, to see it rather than infer it."""
    root = args.out or NORMALIZED_DIR
    if not store.exists(root, args.engine, args.document):
        raise SystemExit(f"no cached text for {args.document!r} under {args.engine!r}")
    result = store.read(root, args.engine, args.document)
    print(f"--- {args.document}  ({args.engine})")
    print(f"  {result.provenance()}")
    print()
    print(result.text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="normalize", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    go = sub.add_parser("run", help="normalize a corpus into the cache")
    go.add_argument("--engine", default="", help="which normalizer (see `engines`)")
    go.add_argument("--corpus", default="", help="corpus root to read")
    go.add_argument("--out", default="", help="cache root to write")
    go.add_argument("--only", default="", help="comma-separated label stems")
    go.add_argument("--limit", type=int, default=0, help="first N documents per type")
    go.add_argument("--concurrency", type=int, default=4)
    go.add_argument("--force", action="store_true", help="redo documents already cached")
    go.add_argument("--dry-run", action="store_true", help="list the work, read nothing")
    go.add_argument("--config", default="")

    ls = sub.add_parser("engines", help="show every normalizer and its settings")
    ls.add_argument("--config", default="")

    one = sub.add_parser("show", help="print the cached text for one document")
    one.add_argument("document", help="corpus-relative path")
    one.add_argument("--engine", default="tesseract")
    one.add_argument("--out", default="")

    args = parser.parse_args(argv)
    # Only `run` takes --only; the other subcommands have no such attribute.
    only = getattr(args, "only", "")
    if only:
        args.only = [part.strip() for part in only.split(",") if part.strip()]
    elif hasattr(args, "only"):
        args.only = None

    if args.command == "run":
        return run(args)
    if args.command == "engines":
        return engines(args)
    if args.command == "show":
        return show(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
