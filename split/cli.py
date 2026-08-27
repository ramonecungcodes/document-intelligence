"""Run a splitter over the bundle corpus and score it against the recorded boundaries.

    python -m split.cli run --splitter every_page
    python -m split.cli run --splitter by_type --limit 20
    python -m split.cli engines

Both baselines are reported alongside whatever was asked for, because a splitter's F1
is unreadable without them: `every_page` has perfect recall by construction and
`single` finds nothing, so a real splitter has to sit above both or it is not earning
its latency.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from core import config as config_mod
from eval.splitting import SplitScore, render
from split.base import SPLITTERS, build

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "/data")
REPORTS_DIR = os.environ.get("DI_REPORTS_DIR", "/reports")


def bundles(corpus_root: str, limit: int):
    path = os.path.join(corpus_root, "labels", "bundles.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"no bundle corpus at {path}.\n"
            f"  Build one first:  docker compose run --rm document-generator bundle")
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    return records[:limit] if limit else records


def score_one(splitter, records, corpus_root: str) -> SplitScore:
    score = SplitScore()
    for record in records:
        result = splitter.split(os.path.join(corpus_root, record["file"]))
        score.add(result.boundaries, record["boundaries"],
                  [d["doc_type"] for d in record["documents"]], result.seconds)
    return score


def run(args) -> int:
    config = config_mod.load(args.config)
    corpus_root = args.corpus or CORPUS_ROOT
    records = bundles(corpus_root, args.limit)
    chosen = (args.splitter or config.chosen("splitter") or "single").strip().lower()

    print(f"{len(records)} bundles  ·  "
          f"{sum(len(r['documents']) for r in records)} documents  ·  "
          f"{sum(r['pages'] for r in records)} pages")

    names = ["single", "every_page"] + ([chosen] if chosen not in
                                        ("single", "every_page") else [])
    results = {}
    for name in names:
        splitter = build(config=config, plugin=name)
        print(f"\n--- {splitter.describe()}", flush=True)
        results[name] = score_one(splitter, records, corpus_root)
        print(render(results[name], name))

    if args.out:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, os.path.basename(args.out))
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"corpus": corpus_root,
                       "scores": {k: v.to_dict() for k, v in results.items()}},
                      handle, indent=1)
        print(f"\nwrote {path}")
    return 0


def engines(args) -> int:
    from core.plugins import describe
    config = config_mod.load(args.config)
    chosen = config.chosen("splitter") or "single"
    for name, cls in sorted(SPLITTERS.items()):
        mark = "  <- selected" if name == chosen else ""
        print(f"--- {name}{mark}")
        print(describe(name, cls.SETTINGS))
        print()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="split")
    sub = parser.add_subparsers(dest="command", required=True)

    go = sub.add_parser("run", help="split the bundle corpus and score it")
    go.add_argument("--splitter", default="")
    go.add_argument("--corpus", default="")
    go.add_argument("--limit", type=int, default=0)
    go.add_argument("--out", default="", help="write the scores as JSON")
    go.add_argument("--config", default="")

    ls = sub.add_parser("engines", help="show every splitter and its settings")
    ls.add_argument("--config", default="")

    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "engines":
        return engines(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
