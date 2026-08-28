#!/usr/bin/env python3
"""Score a classifier over the documents another classifier's report already names.

    python tools/eval-classifier.py --like reports/dit-template.json \
        --classifier cascade --dit-model models/dit-template \
        --out reports/cascade-template.json

The point is the holdout. `tools/train-layout-classifier.py` writes a report whose
`documents` list is a *held-out* set -- and when it was built with `--holdout template`,
held out by page design, which is the only split in this project that does not let a
model recognise a layout it trained on. That split is what turned an apparent 1.000 on
faxes into 0.792, so any number meant to predict a vendor template nobody has seen has
to be measured on it.

Reusing the list rather than re-deriving it is deliberate. The holdout is chosen by a
shuffle inside the training tool, and a second implementation of "which documents were
held out" would drift from the first without either looking wrong -- and the drift would
leak training documents into a test set, which is the one error this whole apparatus
exists to prevent.

The output is the same shape the training tool writes, so `eval.cli calibrate --report`
reads it without knowing which tool produced it.

**--dit-model matters and is not optional in spirit.** A cascade whose primary is the
source-holdout checkpoint, evaluated on design-held-out documents, has seen those
designs in training. It will score well and the number will be worthless. Point it at
the checkpoint that was trained with the same holdout as the report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config as config_mod                                   # noqa: E402

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "data")


def build_normalizer(name: str, corpus: str):
    """The reader the classifier's text arbiter gets, across mixed profiles.

    A design holdout spans clean and degraded documents together, and no single engine
    serves both: the clean ones have an exact text layer and the degraded ones have
    none. `native,cached` reads each correctly -- native wins outright on a real text
    layer, and anything without one falls through to the OCR already on disk.

    Built here rather than through environment overrides on purpose. The normalizer
    cascade and the classifier cascade are both called `cascade` and both declare
    `escalate_below`, so `DI_CASCADE_ESCALATE_BELOW` would silently reach into both
    stages at once. Naming the stack explicitly avoids a variable that means two things.
    """
    from normalize.base import NORMALIZERS, build

    if not name:
        return None
    declares = {s.name for s in NORMALIZERS.get(
        name, type("x", (), {"SETTINGS": ()})).SETTINGS}
    overrides = {"corpus": corpus} if "corpus" in declares else None
    return build(plugin=name, overrides=overrides)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--like", required=True,
                        help="a classifier report whose held-out document list to reuse")
    parser.add_argument("--classifier", default="cascade")
    parser.add_argument("--dit-model", default="", dest="dit_model",
                        help="checkpoint for the cascade's image primary; must share "
                             "the holdout of --like")
    parser.add_argument("--normalizer", default="cascade",
                        help="reader for the text arbiter (default: %(default)s, "
                             "which is native then the OCR cache)")
    parser.add_argument("--corpus", default=CORPUS_ROOT)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    with open(args.like, encoding="utf-8") as handle:
        source = json.load(handle)
    documents = source.get("documents") or []
    if args.limit:
        documents = documents[:args.limit]
    if not documents:
        raise SystemExit(f"{args.like} names no documents")
    holdout = source.get("holdout")
    print(f"  reusing {len(documents)} documents held out by {holdout!r}")
    if holdout == "template" and args.dit_model and "template" not in args.dit_model:
        # Not fatal -- a deliberate cross-holdout comparison is a real thing to want --
        # but it is never what somebody means by accident.
        print(f"  WARNING: holdout is 'template' but the checkpoint is "
              f"{args.dit_model!r}. If that checkpoint was trained on these designs, "
              f"the score below is meaningless.", file=sys.stderr)

    config = config_mod.load(args.config)
    if args.dit_model:
        os.environ["DI_DIT_MODEL"] = args.dit_model

    from classify.base import build as build_classifier

    classifier = build_classifier(config=config, plugin=args.classifier,
                                  overrides={"normalizer": args.normalizer}
                                  if args.normalizer else None)
    print(f"  classifier: {classifier.describe()}")

    normalizer = None
    if getattr(classifier, "NEEDS_TEXT", True):
        normalizer = build_normalizer(args.normalizer, args.corpus)
        print(f"  normalizer: {normalizer.describe()}")

    rows, started, missing = [], time.time(), 0
    for index, entry in enumerate(documents, 1):
        relative = str(entry.get("file", "")).replace("\\", "/")
        # Reports carry the corpus-rooted path; the corpus root is passed separately.
        path = relative
        if not os.path.exists(path):
            path = os.path.join(args.corpus, relative)
        if not os.path.exists(path):
            missing += 1
            continue
        document = normalizer.read(path) if normalizer else None
        result = classifier.classify(document.text if document else "",
                                     document=document, path=path,
                                     corpus=args.corpus)
        rows.append({
            "file": relative,
            "profile": entry.get("profile"),
            "truth": entry.get("truth"),
            # What it would have answered, before any floor. A report recording only
            # the post-floor answer cannot be used to ask where the floor belongs.
            "predicted": result.withheld or result.label,
            "confidence": result.confidence,
            "margin": result.margin,
            "runner_up": result.runner_up,
            "abstained": result.abstained,
            "engine": result.engine,
            "escalated": "escalated" in (result.evidence or ""),
        })
        if index % 25 == 0 or index == len(documents):
            print(f"    {index}/{len(documents)}   "
                  f"{index / max(1e-6, time.time() - started):.1f}/s")

    if missing:
        print(f"  {missing} documents named in {args.like} are not on disk",
              file=sys.stderr)

    correct = sum(1 for r in rows if r["predicted"] == r["truth"])
    escalated = sum(1 for r in rows if r["escalated"])
    report = {
        "classifier": classifier.describe(),
        "normalizer": normalizer.describe() if normalizer else None,
        "holdout": holdout,
        "like": args.like,
        "dit_model": args.dit_model or None,
        "documents_scored": len(rows),
        "accuracy": round(correct / len(rows), 4) if rows else None,
        "escalated": escalated,
        "escalation_rate": round(escalated / len(rows), 4) if rows else None,
        "documents": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=1)
    print(f"  accuracy {report['accuracy']}   escalated "
          f"{escalated}/{len(rows)} ({report['escalation_rate']})")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
