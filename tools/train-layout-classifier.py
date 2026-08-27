"""Fine-tune a layout-aware classifier, and measure it against the text one.

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
LABELS = sorted(set(TYPE_OF.values()))
PROFILES = ("light", "photo", "fax")


def held_out(repo: str) -> set:
    """The source documents behind the 75-document stratified sample."""
    path = os.path.join(repo, "data", "sample75.txt")
    with open(path, encoding="utf-8") as handle:
        return {line.strip().split("__")[0] for line in handle if line.strip()}


def catalogue(repo: str):
    """Every (path, words, label, profile, source) the training set can draw on.

    Degraded documents read their words from the docTR cache; clean ones read theirs
    from the PDF's own text layer, which is exact and free. A document whose OCR has
    not been run yet is skipped rather than silently read some other way -- a training
    set that quietly mixes engines is one whose result cannot be attributed.
    """
    from normalize.native import NativeText

    native = NativeText(keep_words=True)
    cache = os.path.join(repo, "data", "normalized")
    items, missing = [], 0

    for folder, label in TYPE_OF.items():
        clean_dir = os.path.join(repo, "data", folder)
        if not os.path.isdir(clean_dir):
            continue
        for name in sorted(os.listdir(clean_dir)):
            if not name.endswith(".pdf"):
                continue
            source = f"{folder}/{name[:-4]}"
            items.append({"path": os.path.join(clean_dir, name), "label": label,
                          "profile": "clean", "source": source, "words": None,
                          "reader": "native"})
            for profile in PROFILES:
                key = f"{folder}/{name[:-4]}__{profile}.pdf"
                degraded = os.path.join(repo, "data", "degraded", key)
                if not os.path.exists(degraded):
                    continue
                if not store.exists(cache, "doctr", key):
                    missing += 1
                    continue
                items.append({"path": degraded, "label": label, "profile": profile,
                              "source": source, "words": key, "reader": "doctr"})
    return items, missing, native


def build_features(items, native, repo: str):
    """Words, boxes and a page image per document."""
    built = []
    for index, item in enumerate(items, 1):
        if item["reader"] == "native":
            words = native.read(item["path"]).words
        else:
            words = store.read(os.path.join(repo, "data", "normalized"),
                               "doctr", item["words"]).words
        texts, boxes, image = features(item["path"], words)
        if not texts:
            # No readable words at all. Kept out rather than taught as an empty
            # example of its class, which would teach that a blank page is that type.
            continue
        built.append({**item, "texts": texts, "boxes": boxes, "image": image})
        if index % 200 == 0:
            print(f"  features {index}/{len(items)}", flush=True)
    return built


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
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoProcessor, AutoModelForSequenceClassification

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo = os.path.abspath(args.repo)
    test_sources = held_out(repo)
    items, missing, native = catalogue(repo)
    if missing:
        print(f"note: {missing} degraded documents have no docTR text yet; skipped")

    train_items = [i for i in items if i["source"] not in test_sources]
    test_items = [i for i in items if i["source"] in test_sources]
    if args.limit:
        random.shuffle(train_items)
        train_items = train_items[:args.limit]
    print(f"{len(train_items)} training - {len(test_items)} held out "
          f"({len(test_sources)} source documents) - device {device}")

    print("building features ...", flush=True)
    train = build_features(train_items, native, repo)
    test = build_features(test_items, native, repo)
    print(f"  {len(train)} train - {len(test)} test")

    processor = AutoProcessor.from_pretrained(args.checkpoint, apply_ocr=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)}).to(device)

    def encode(batch):
        encoded = processor(
            [b["image"] for b in batch], [b["texts"] for b in batch],
            boxes=[b["boxes"] for b in batch], truncation=True, padding=True,
            max_length=512, return_tensors="pt")
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

    report = os.path.join(repo, "reports", "layout-classifier.json")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"checkpoint": args.checkpoint, "epochs": args.epochs,
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
