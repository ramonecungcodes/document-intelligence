#!/usr/bin/env python3
"""Re-derive the classifier signals for a run made before they were recorded.

    python tools/recover-signals.py --predictions reports/v1-predicted-type.jsonl
    python tools/recover-signals.py --predictions reports/v1-predicted-type.jsonl \
        --classifier cascade --corpus data

Runs the classifier again over the documents an existing predictions file names, and
writes the `.signals.jsonl` sidecar that run never produced. It does not touch the
predictions.

**This is only valid because the classifier is deterministic and the checkpoint has not
changed.** DiT in eval mode with a fixed checkpoint returns the same probabilities for
the same page every time, so re-running it reproduces exactly what was said during the
original run rather than approximating it. Two things break that guarantee and both
make the output a fabrication rather than a recovery:

    the checkpoint has been retrained or replaced since the run
    the manifest now names a different classifier, or different settings

Neither is detectable from here -- a checkpoint directory carries no record of which
run used it -- so the recovered file records what it was produced with, and the
comparison is left to a person who knows what changed. This is why the signals are
written during the run in the first place. Recovery is the fallback, not the design.

An extraction cannot be recovered this way at all: the model is remote, sampled, and
not pinned. Only signals from deterministic stages belong here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config as config_mod                                   # noqa: E402
from eval import score as scoring                                       # noqa: E402
from route import signals as signals_mod                                # noqa: E402

CORPUS_ROOT = os.environ.get("DI_DATASET_ROOT", "data")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True,
                        help="the predictions file whose sidecar is missing")
    parser.add_argument("--corpus", default=CORPUS_ROOT)
    parser.add_argument("--config", default=None, help="manifest (default: di.toml)")
    parser.add_argument("--classifier", default="",
                        help="override the manifest's choice")
    parser.add_argument("--normalizer", default="",
                        help="override the manifest's choice")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many documents, for a smoke test")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing sidecar. Refused by default: a "
                             "sidecar written during the run is the real record and "
                             "a recovered one must never silently replace it")
    args = parser.parse_args(argv)

    out_path = signals_mod.path_for(args.predictions)
    if os.path.exists(out_path) and not args.force:
        raise SystemExit(
            f"{out_path} already exists.\n"
            f"  A sidecar written during the run is the record; this tool only "
            f"reconstructs one.\n"
            f"  Pass --force if you are certain you want to replace it.")

    if not os.path.exists(args.predictions):
        raise SystemExit(f"predictions file not found: {args.predictions}")
    records = scoring.load_records(args.predictions)
    files = []
    for record in records:
        name = str(record.get("file", "")).replace("\\", "/")
        if name:
            files.append(name)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no `file` keys in {args.predictions}")

    config = config_mod.load(args.config)
    from classify.base import build as build_classifier

    overrides = {"normalizer": args.normalizer} if args.normalizer else None
    classifier = build_classifier(config=config, plugin=args.classifier,
                                  overrides=overrides)
    print(f"  classifier: {classifier.describe()}")

    # The same test the runner applies. A classifier that reads only pixels must not
    # drag an OCR pass over the corpus behind it just because this is a recovery.
    normalizer = None
    if getattr(classifier, "NEEDS_TEXT", True):
        from normalize.base import NORMALIZERS, build as build_normalizer
        chosen = (config.chosen("normalizer", args.normalizer)
                  or "native").strip().lower()
        declares = {x.name for x in NORMALIZERS.get(
            chosen, type("x", (), {"SETTINGS": ()})).SETTINGS}
        normalizer = build_normalizer(
            config=config, plugin=args.normalizer,
            overrides={"corpus": args.corpus} if "corpus" in declares else None)
        print(f"  normalizer: {normalizer.describe()}")

    writer = signals_mod.Writer(args.predictions)
    started = time.time()
    missing = 0
    for index, rel in enumerate(files, 1):
        path = os.path.join(args.corpus, rel)
        if not os.path.exists(path):
            missing += 1
            continue
        document = normalizer.read(path) if normalizer else None
        result = classifier.classify(document.text if document else "",
                                     document=document, path=path,
                                     corpus=args.corpus)
        writer.record(rel,
                      classifier=signals_mod.from_classification(result),
                      normalizer=signals_mod.from_normalizer(document))
        if index % 25 == 0 or index == len(files):
            rate = index / max(1e-6, time.time() - started)
            print(f"    {index}/{len(files)}   {rate:.1f}/s")

    if missing:
        print(f"  {missing} documents named in the predictions are not in the corpus",
              file=sys.stderr)
    path = writer.write()
    # Stamped so the file cannot later be mistaken for one written during the run.
    # Without this the two are byte-identical in shape and only the mtime distinguishes
    # them, which is not a distinction anybody checks.
    stamp = out_path + ".provenance.json"
    with open(stamp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({
            "recovered": True,
            "predictions": args.predictions,
            "classifier": classifier.describe(),
            "normalizer": normalizer.describe() if normalizer else None,
            "manifest": args.config or "di.toml",
            "documents": len(writer),
            "note": "Re-derived after the fact by tools/recover-signals.py. Valid only "
                    "while the checkpoint and settings match the original run; nothing "
                    "here can verify that they do.",
        }, handle, indent=1)
    print(f"  wrote {path}   ({len(writer)} documents)")
    print(f"  wrote {stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
