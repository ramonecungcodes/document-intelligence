"""Run a classifier over a corpus and score it against the labels.

    python -m classify.cli run --classifier keyword
    python -m classify.cli run --classifier llm --limit 20
    python -m classify.cli run --classifier llm --corpus /data/degraded
    python -m classify.cli engines

Unlike extraction, classification is cheap enough to score in the same pass -- one
label per document, compared against one truth. There is no join to get wrong, so the
report comes out of the same command that produced it.

The baseline is always reported alongside. With five types and a skewed corpus, always
answering `form` scores 0.45 having read nothing, and a classifier that cannot beat
grep is not earning its latency.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from classify.base import CLASSIFIERS, build
from core import config as config_mod
from core import doctypes
from eval.classification import ClassificationScore, render
from eval.score import load_corpus
from extract.cli import sample

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "/data")
REPORTS_DIR = os.environ.get("DI_REPORTS_DIR", "/reports")


def documents(corpus_root, only, limit):
    """(relative path, true type) for every document to classify."""
    found = []
    for stem, records in load_corpus(corpus_root, only).items():
        doctype = doctypes.for_label_file(stem)
        if doctype is None:
            continue
        for record in sample(records, doctype, limit):
            found.append((record["file"], record["doc_type"]))
    return found


def run(args) -> int:
    config = config_mod.load(args.config)
    classifier = build(config=config, plugin=args.classifier)
    corpus_root = args.corpus or CORPUS_ROOT

    # Text comes from the normalizer, so a classifier can be measured on degraded
    # documents without knowing OCR happened -- the same seam the extractor uses.
    from normalize.base import NORMALIZERS, build as build_normalizer
    chosen = (config.chosen("normalizer", args.normalizer) or "native").strip().lower()
    declares = {s.name for s in NORMALIZERS.get(
        chosen, type("x", (), {"SETTINGS": ()})).SETTINGS}
    normalizer = build_normalizer(
        config=config, plugin=args.normalizer,
        overrides={"corpus": corpus_root} if "corpus" in declares else None)

    jobs = documents(corpus_root, args.only, args.limit)
    if not jobs:
        raise SystemExit("nothing to classify")
    print(f"{len(jobs)} documents · {classifier.describe()} · "
          f"text from {normalizer.describe()}")

    score = ClassificationScore()
    rows = []

    def work(job):
        relative, truth = job
        path = os.path.join(corpus_root, relative)
        document = normalizer.read(path)
        # The whole `Extracted` and the path, not just the text. A layout-aware
        # classifier needs the word boxes and the page image; the text ones declare
        # **_ and ignore both, so nothing else had to change to allow it.
        return relative, truth, classifier.classify(
            document.text, document=document, path=path)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                relative, truth, result = future.result()
            except Exception as error:
                print(f"  FAILED: {type(error).__name__}: {error}", file=sys.stderr)
                continue
            score.add(truth, result.doc_type, result.runner_up, result.seconds)
            rows.append({"file": relative, "truth": truth,
                         "predicted": result.doc_type,
                         **result.provenance()})
            if done % 25 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}]", flush=True)

    print(render(score))

    if args.out:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, os.path.basename(args.out))
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"classifier": classifier.describe(),
                       "corpus": corpus_root,
                       "score": score.to_dict(),
                       "documents": rows}, handle, indent=1)
        print(f"\nwrote {path}")
    return 0


def engines(args) -> int:
    from core.plugins import describe
    config = config_mod.load(args.config)
    chosen = config.chosen("classifier") or "keyword"
    for name, cls in sorted(CLASSIFIERS.items()):
        mark = "  <- selected" if name == chosen else ""
        print(f"--- {name}{mark}")
        print(describe(name, cls.SETTINGS))
        print()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="classify")
    sub = parser.add_subparsers(dest="command", required=True)

    go = sub.add_parser("run", help="classify a corpus and score it")
    go.add_argument("--classifier", default="", help="keyword | llm")
    go.add_argument("--normalizer", default="", help="how text is obtained")
    go.add_argument("--corpus", default="")
    go.add_argument("--only", default="")
    go.add_argument("--limit", type=int, default=0)
    go.add_argument("--concurrency", type=int, default=4)
    go.add_argument("--out", default="", help="write the full result as JSON")
    go.add_argument("--config", default="")

    ls = sub.add_parser("engines", help="show every classifier and its settings")
    ls.add_argument("--config", default="")

    args = parser.parse_args(argv)
    only = getattr(args, "only", "")
    if hasattr(args, "only"):
        args.only = [p.strip() for p in only.split(",") if p.strip()] or None

    if args.command == "run":
        return run(args)
    if args.command == "engines":
        return engines(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
