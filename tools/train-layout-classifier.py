"""Fine-tune a document classifier that reads the page, and measure it against text.

Two architectures, one harness:

    --arch layout   LayoutLMv3 -- words, word boxes and the page image
    --arch image    DiT -- the page image alone, no words and no boxes

They share the split, the held-out set, the scorer and the report on purpose. The
comparison between them is the point, and two scripts would eventually disagree about
what they were comparing.

Why this exists
---------------
Phase 3's LLM classifier scores 0.990 on clean documents and 0.571 on faxes. The gap
is not comprehension -- it is that docTR loses 38% of the words on a 170 dpi bitonal
page, and a text-only classifier cannot read what OCR never found.

A layout-aware model reads word *positions* and the page image as well as the words,
and geometry degrades more gracefully than glyphs do. That was checked before any of
this was built: nearest neighbour over a coarse ink-occupancy grid, no words at all
and nothing trained, already scored 0.821 on the same faxes. This asks whether a model
that learns class-level layout does better, on held-out documents, where the templates
it saw in training are not the templates it is tested on.

The split is the part worth reading
-----------------------------------
Held out by *source document*, not by file. Every document exists four times -- clean,
light, photo, fax -- and putting a document's light version in training while its fax
version is in the test set would be measuring memorisation of that document, which is
exactly the thing this is supposed to distinguish from learning a layout class.

The held-out set is the same 75-document stratified sample Phase 2 and Phase 3 were
scored on. That is not convenience: it makes the fax number here directly comparable
to the LLM's 0.571 on the same fax documents, rather than to a number from a different
sample that would have to be caveated into uselessness.

Training text comes from docTR for degraded documents and the embedded text layer for
clean ones -- each the best available reading of that document, which is what the
pipeline would do in production too.

Licensing
---------
`microsoft/layoutlmv3-base` is CC-BY-NC-SA 4.0: fine for a portfolio, not for a
commercial deployment. `--checkpoint` exists so the model can be swapped; LiLT
(`SCUT-DLVCLab/lilt-roberta-en-base`) is MIT and takes the same text-and-boxes input.
Check `microsoft/dit-base`'s own terms before shipping it -- do not assume it inherits
the MIT licence of the unilm repository it is published from.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classify.features import features                     # noqa: E402
from normalize import store                                # noqa: E402

TYPE_OF = {"forms": "form", "invoices": "invoice",
           "multi_bill_invoices": "multi_bill_invoice",
           "purchase_orders": "purchase_order", "resumes": "resume"}
# Nine, not five. `form` covers five variants whose field sets differ by more than a
# name -- onboarding asks for 22 fields, w4 for 9 -- and the extractor picks between
# them. Training on `form` would leave that choice to the corpus, which is the
# hand-over this phase exists to remove. The list comes from the type registry so the
# classifier and the extractor cannot disagree about what the answers are.
from classify.base import labels as _labels, split_label     # noqa: E402

LABELS = list(_labels())
PROFILES = ("light", "photo", "fax")


def held_out(repo: str, per_label: int, seed: int) -> set:
    """An equal number of source documents reserved per label.

    The old held-out set was a stratified sample built for a different purpose, and it
    was stratified by document *type* rather than by label: it reserved 32 resumes and
    four of each form variant. A test set that is 28% one class makes the headline
    number a statement about that class.

    Chosen by source document, because every document exists four times -- clean,
    light, photo, fax -- and reserving a document's fax version while training on its
    light version would measure memorisation of that document rather than of its type.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for folder, base in TYPE_OF.items():
        clean_dir = os.path.join(repo, "data", folder)
        if not os.path.isdir(clean_dir):
            continue
        variants = variant_index(repo)
        for name in sorted(os.listdir(clean_dir)):
            if not name.endswith(".pdf"):
                continue
            source = f"{folder}/{name[:-4]}"
            variant = variants.get(source, "")
            grouped[f"{base}:{variant}" if variant else base].append(source)
    # Spread across page designs, not drawn blind. Taking four invoices at random from
    # ten designs lands on three distinct ones and tests a duplicate instead of a
    # design -- the variety is in the corpus and the test set should see it. Round-robin
    # over designs first, then over documents within a design.
    by_design = defaultdict(lambda: defaultdict(list))
    designs = layouts(repo)
    for label, sources in grouped.items():
        for source in sources:
            by_design[label][designs.get(source, "")].append(source)

    rng = random.Random(seed)
    chosen = set()
    for label in sorted(grouped):
        buckets = []
        # Design order is shuffled as well as document order. Walking them in sorted
        # order would hold out designs 0-3 of every type on every run and leave 4-9
        # never evaluated -- distinct, but always the same quarter of the space.
        for design in sorted(by_design[label]):
            pool = sorted(by_design[label][design])
            rng.shuffle(pool)
            buckets.append(pool)
        rng.shuffle(buckets)
        picked, depth = [], 0
        while len(picked) < per_label and any(len(b) > depth for b in buckets):
            for bucket in buckets:
                if len(bucket) > depth and len(picked) < per_label:
                    picked.append(bucket[depth])
            depth += 1
        chosen.update(picked)
    return chosen


def layouts(repo: str):
    """file stem -> which visual template the generator drew it from.

    The corpus is generated, so every document of a type shares one of a handful of
    designs. Holding out documents cannot tell a model that has learned what an
    invoice is from one that has memorised what this corpus's invoice template looks
    like -- the held-out document is drawn from a template the model trained on.

    Only forms, multi-bill invoices and resumes carry a `layout`. Invoices and purchase
    orders have exactly one design each, so no split of this corpus can ask whether a
    model generalises across invoice designs. That is a limit of the corpus and it is
    reported rather than papered over.
    """
    import glob
    found = {}
    for path in glob.glob(os.path.join(repo, "data", "labels", "*.json")):
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        records = records if isinstance(records, list) else (
            records.get("documents") or records.get("records") or [])
        for record in records:
            if record.get("layout") is None or not record.get("file"):
                continue
            stem = record["file"].replace("\\", "/")
            found[stem[:-4] if stem.endswith(".pdf") else stem] = str(record["layout"])
    return found


def unseen_templates(by_layout):
    """Reserve the highest-numbered template of each type for the test set."""
    from collections import defaultdict
    grouped = defaultdict(set)
    for stem, layout in by_layout.items():
        grouped[stem.split("/")[0]].add(layout)
    return {folder: max(seen) for folder, seen in grouped.items()}


def variant_index(repo: str) -> dict:
    """file stem -> the variant its own label file records, for types that have them."""
    import glob
    from core import doctypes
    found = {}
    for path in glob.glob(os.path.join(repo, "data", "labels", "*.json")):
        stem = os.path.basename(path)[:-5]
        doctype = doctypes.for_label_file(stem)
        if doctype is None or not doctype.variant_key:
            continue
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        records = records if isinstance(records, list) else (
            records.get("documents") or records.get("records") or [])
        for record in records:
            variant = doctype.variant_of(record)
            key = record.get("file", "").replace("\\", "/")
            if variant and key:
                found[key[:-4] if key.endswith(".pdf") else key] = variant
    return found


def catalogue(repo: str, require_ocr: bool = True):
    """Every (path, words, label, profile, source) the training set can draw on.

    Degraded documents read their words from the docTR cache; clean ones read theirs
    from the PDF's own text layer, which is exact and free. A document whose OCR has
    not been run yet is skipped rather than silently read some other way -- a training
    set that quietly mixes engines is one whose result cannot be attributed.

    Except for the image architecture, which reads no words at all. Making DiT wait on
    an OCR pass it never consults would have quietly trained it on the 352 clean
    documents alone and called the result a degradation number.
    """
    from normalize.native import NativeText

    native = NativeText(keep_words=True)
    cache = os.path.join(repo, "data", "normalized")
    variants = variant_index(repo)
    items, missing = [], 0

    for folder, _base_type in TYPE_OF.items():
        clean_dir = os.path.join(repo, "data", folder)
        if not os.path.isdir(clean_dir):
            continue
        for name in sorted(os.listdir(clean_dir)):
            if not name.endswith(".pdf"):
                continue
            source = f"{folder}/{name[:-4]}"
            variant = variants.get(source, "")
            label = f"{TYPE_OF[folder]}:{variant}" if variant else TYPE_OF[folder]
            items.append({"path": os.path.join(clean_dir, name), "label": label,
                          "profile": "clean", "source": source, "words": None,
                          "reader": "native"})
            for profile in PROFILES:
                key = f"{folder}/{name[:-4]}__{profile}.pdf"
                degraded = os.path.join(repo, "data", "degraded", key)
                if not os.path.exists(degraded):
                    continue
                if require_ocr and not store.exists(cache, "doctr", key):
                    missing += 1
                    continue
                items.append({"path": degraded, "label": label, "profile": profile,
                              "source": source, "words": key, "reader": "doctr"})
    return items, missing, native


def build_features(items, native, repo: str, require_words: bool = True):
    """Words, boxes and a page image per document.

    `require_words` is False for the image-only architecture, and the distinction is
    not cosmetic. A fax page docTR read three words from still has a perfectly good
    picture, and dropping it would remove the hardest documents from the image model's
    training *and* its test set -- handing it a flattering score against a model that
    had to face them.
    """
    built = []
    for index, item in enumerate(items, 1):
        if item["reader"] == "native" or not require_words:
            # The image architecture never looks at these; reading OCR JSON for a
            # thousand documents to hand it an unused list is pure latency.
            words = native.read(item["path"]).words if item["reader"] == "native" else []
        else:
            words = store.read(os.path.join(repo, "data", "normalized"),
                               "doctr", item["words"]).words
        texts, boxes, image = features(item["path"], words)
        if require_words and not texts:
            # No readable words at all. Kept out of the text-and-boxes model rather
            # than taught as an empty example of its class, which would teach it that
            # a blank page is that type.
            continue
        built.append({**item, "texts": texts, "boxes": boxes, "image": image})
        if index % 200 == 0:
            print(f"  features {index}/{len(items)}", flush=True)
    return built


def balance(items, seed: int):
    """Cap every label to the rarest, so the test set cannot be read as one class.

    Reserving a page design takes a third of the forms and a tenth of the invoices,
    because forms have three designs and invoices have ten. Left alone that hands back
    a test set which is 74% forms, where a model that answered `form` every time would
    score 0.74 and the headline number would say almost nothing about the other four
    types.

    Capping is deterministic and it is announced. A silently truncated evaluation set
    reads as full coverage, which is the failure this whole project is arranged to
    avoid.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for item in items:
        # Keyed on label *and* profile. Capping on the label alone could keep four
        # faxes for one class and four clean pages for another, and the per-profile
        # table -- the one that says whether degradation is handled -- would be
        # comparing different classes at each row.
        grouped[(item["label"], item["profile"])].append(item)
    if not grouped:
        return items
    cap = min(len(v) for v in grouped.values())
    kept, dropped = [], 0
    rng = random.Random(seed)
    for label in sorted(grouped):
        pool = sorted(grouped[label], key=lambda i: i["path"])
        rng.shuffle(pool)
        kept += pool[:cap]
        dropped += len(pool) - cap
    if dropped:
        print(f"  test set balanced to {cap} per label and profile "
              f"({len(kept)} kept, {dropped} dropped so no class dominates)")
    return kept


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="train-layout-classifier")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--checkpoint", default="microsoft/layoutlmv3-base")
    parser.add_argument("--out", default="models/layout-classifier")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=0, help="smoke test on a subset")
    parser.add_argument("--balance", action="store_true",
                        help="weight the loss by inverse class frequency")
    parser.add_argument("--profiles", default="",
                        help="restrict to these profiles, e.g. 'clean'. Clean documents "
                             "carry an exact text layer, so a words-and-boxes model can "
                             "be measured on them without an OCR pass at all.")
    parser.add_argument("--test-per-label", type=int, default=4,
                        help="source documents reserved per label (holdout=source)")
    parser.add_argument("--unbalanced-test", action="store_true",
                        help="do not cap the test set to equal counts per label")
    parser.add_argument("--holdout", default="source", choices=("source", "template"),
                        help="source: unseen documents. template: unseen page designs")
    parser.add_argument("--arch", default="layout", choices=("layout", "image"),
                        help="layout: words+boxes+image (LayoutLMv3). "
                             "image: the page picture alone (DiT)")
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                              AutoProcessor, AutoModelForSequenceClassification)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo = os.path.abspath(args.repo)
    test_sources = held_out(repo, args.test_per_label, args.seed)
    items, missing, native = catalogue(repo, require_ocr=args.arch != "image")
    if missing:
        print(f"note: {missing} degraded documents have no docTR text yet; skipped")

    if args.holdout == "template":
        by_layout = layouts(repo)
        reserved = unseen_templates(by_layout)
        print("  reserved templates: " + "  ".join(
            f"{k}=layout {v}" for k, v in sorted(reserved.items())))
        def is_test(item):
            layout = by_layout.get(item["source"])
            return layout is not None and reserved.get(
                item["source"].split("/")[0]) == layout
        untestable = sorted({i["source"].split("/")[0] for i in items
                             if by_layout.get(i["source"]) is None})
        if untestable:
            print(f"  no template variation, so untestable this way: "
                  f"{', '.join(untestable)}")
    else:
        def is_test(item):
            return item["source"] in test_sources

    if args.profiles:
        want = {p.strip() for p in args.profiles.split(",") if p.strip()}
        items = [i for i in items if i["profile"] in want]
        print(f"  restricted to profiles: {', '.join(sorted(want))}")

    train_items = [i for i in items if not is_test(i)]
    test_items = [i for i in items if is_test(i)]
    if not args.unbalanced_test:
        # Dropped documents go nowhere. They cannot join training -- they are drawn
        # from a reserved design, and putting them back would dissolve the holdout
        # this split exists to create.
        test_items = balance(test_items, args.seed)
    if args.limit:
        random.shuffle(train_items)
        train_items = train_items[:args.limit]
    print(f"{len(train_items)} training - {len(test_items)} held out "
          f"({len(test_sources)} source documents) - device {device}")

    print("building features ...", flush=True)
    needs_words = args.arch != "image"
    train = build_features(train_items, native, repo, needs_words)
    test = build_features(test_items, native, repo, needs_words)
    print(f"  {len(train)} train - {len(test)} test")

    ids = {"id2label": {i: l for i, l in enumerate(LABELS)},
           "label2id": {l: i for i, l in enumerate(LABELS)}}
    if args.arch == "image":
        # DiT sees the page and nothing else -- no words, no boxes. It is the honest
        # test of how much the picture alone carries, which the multimodal model
        # cannot answer because it always has the text to fall back on.
        processor = AutoImageProcessor.from_pretrained(args.checkpoint)
        model = AutoModelForImageClassification.from_pretrained(
            args.checkpoint, num_labels=len(LABELS), **ids,
            ignore_mismatched_sizes=True).to(device)
    else:
        processor = AutoProcessor.from_pretrained(args.checkpoint, apply_ocr=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint, num_labels=len(LABELS), **ids).to(device)

    def encode(batch):
        if args.arch == "image":
            encoded = processor([b["image"] for b in batch], return_tensors="pt")
        else:
            encoded = processor(
                [b["image"] for b in batch], [b["texts"] for b in batch],
                boxes=[b["boxes"] for b in batch], truncation=True, padding=True,
                max_length=512, return_tensors="pt")
        encoded = dict(encoded)
        encoded["labels"] = torch.tensor([LABELS.index(b["label"]) for b in batch])
        return {k: v.to(device) for k, v in encoded.items()}

    class Docs(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    counts = {label: sum(1 for t in train if t["label"] == label) for label in LABELS}
    print("  training mix: " + "  ".join(f"{l}={counts[l]}" for l in LABELS))
    weights = None
    if args.balance:
        # The corpus really is skewed -- forms outnumber resumes four to one -- and a
        # model that answers `form` whenever it is unsure scores respectably while
        # having learned very little. Inverse-frequency weighting makes a rare class's
        # mistakes cost what a common one's do. It cannot hide the collapse it is
        # meant to prevent: per-class recall is reported either way.
        total = sum(counts.values())
        weights = torch.tensor(
            [total / (len(LABELS) * max(1, counts[l])) for l in LABELS],
            dtype=torch.float, device=device)
        print("  class weights: " + "  ".join(
            f"{l}={w:.2f}" for l, w in zip(LABELS, weights.tolist())))

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(Docs(train), batch_size=args.batch, shuffle=True,
                        collate_fn=lambda b: b)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    steps = max(1, args.epochs * (len(loader) // args.accumulate))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    print(f"training {args.epochs} epochs, {steps} optimizer steps", flush=True)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader, 1):
            encoded = encode(batch)
            labels = encoded.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**encoded).logits
                loss = loss_fn(logits.float(), labels) / args.accumulate
            scaler.scale(loss).backward()
            total += loss.item() * args.accumulate
            seen += 1
            if step % args.accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if schedule.last_epoch < steps - 1:
                    schedule.step()
        print(f"  epoch {epoch}  loss {total / max(1, seen):.4f}  "
              f"{time.time() - started:.0f}s", flush=True)

    target = os.path.join(repo, args.out)
    os.makedirs(target, exist_ok=True)
    model.save_pretrained(target)
    processor.save_pretrained(target)
    print(f"saved to {args.out}")

    # ---------------------------------------------------------------- evaluation
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(test), args.batch):
            batch = test[i:i + args.batch]
            encoded = encode(batch)
            encoded.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**encoded).logits.float()
            probability = torch.softmax(logits, dim=-1)
            order = probability.argsort(dim=-1, descending=True)
            for j, item in enumerate(batch):
                rows.append({
                    "file": os.path.relpath(item["path"], repo).replace("\\", "/"),
                    "profile": item["profile"], "truth": item["label"],
                    "predicted": LABELS[order[j][0].item()],
                    "confidence": round(probability[j][order[j][0]].item(), 4),
                    "runner_up": LABELS[order[j][1].item()],
                })

    # Named after the model it describes. Two runs writing one filename means the
    # second silently destroys the first's evidence, which in a repository whose
    # argument is its numbers is worse than not writing the file at all.
    report = os.path.join(repo, "reports", f"{os.path.basename(args.out)}.json")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"checkpoint": args.checkpoint, "arch": args.arch,
                   "holdout": args.holdout,
                   "epochs": args.epochs, "balanced": args.balance,
                   "train": len(train), "documents": rows}, handle, indent=1)

    # Scored through the same code the LLM classifier is scored through, so the two
    # numbers are comparable without anyone having to check that two renderers agree
    # about what accuracy means.
    from eval.classification import ClassificationScore, render

    print("\nHELD-OUT ACCURACY BY PROFILE\n")
    print(f"  {'profile':<10}{'n':>5}{'accuracy':>11}{'baseline':>11}")
    for profile in ("clean",) + PROFILES:
        rs = [r for r in rows if r["profile"] == profile]
        if not rs:
            continue
        score = ClassificationScore()
        for r in rs:
            score.add(r["truth"], r["predicted"], r["runner_up"])
        data = score.to_dict()
        print(f"  {profile:<10}{len(rs):>5}{data['accuracy']:>11.3f}"
              f"{data['majority_baseline']:>11.3f}")

    overall = ClassificationScore()
    for r in rows:
        overall.add(r["truth"], r["predicted"], r["runner_up"])
    print(render(overall))
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
