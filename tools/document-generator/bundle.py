#!/usr/bin/env python3
"""Concatenate documents into multi-document PDFs, the way a scanner batch arrives.

The splitter is the one stage of this pipeline that had no corpus at all. Every
document in the set is exactly one document, so a splitter could be written but not
measured, and a stage that cannot be measured is a stage nobody should claim works.

A bundle is several documents in one file, and the labels record where each one starts
and ends. That makes boundary detection scorable the same way everything else here is:
against a truth that was written down before anything ran.

Three properties are deliberate, because a bundle corpus is easy to make too easy.

Same-type adjacency. Two invoices back to back is the case a per-page classifier cannot
see -- the type never changes, so a boundary detector keyed on type change is blind to
it. If bundles were drawn at random from five types those would be a fifth of the
joins; they are forced to roughly half, because the hard case is the one worth
measuring.

Multi-page documents. A two-page form followed by an invoice has an interior page that
is not a boundary, and a splitter that answers "every page is a document" has to be
punished for it. The corpus carries 32 two-page documents and they are preferred when
building bundles rather than avoided.

Singletons. Not every scan holds several documents, and a splitter that always finds a
boundary somewhere should lose points on the files that hold exactly one.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

OUT = os.environ.get("DI_DATASET_ROOT", "/data")
TYPES = ("forms", "invoices", "multi_bill_invoices", "purchase_orders", "resumes")


def open_pdf(path):
    try:
        import pymupdf
    except ImportError:                       # pragma: no cover
        import fitz as pymupdf
    return pymupdf.open(path)


def catalogue(root: str):
    """Every source document with its type, page count and label record."""
    found = []
    for folder in TYPES:
        labels = os.path.join(root, "labels", f"{folder}.json")
        if not os.path.exists(labels):
            continue
        with open(labels, encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            path = os.path.join(root, record["file"])
            if not os.path.exists(path):
                continue
            document = open_pdf(path)
            pages = document.page_count
            document.close()
            found.append({"file": record["file"], "folder": folder,
                          "doc_type": record["doc_type"], "pages": pages,
                          "variant": record.get("form_type") or record.get("layout")})
    return found


def plan(pool, count: int, seed: int, same_type_rate: float, singleton_rate: float):
    """Which documents go in which bundle, before any PDF is touched."""
    rng = random.Random(seed)
    by_type = collections.defaultdict(list)
    for item in pool:
        by_type[item["doc_type"]].append(item)
    for items in by_type.values():
        rng.shuffle(items)
    cursors = {t: 0 for t in by_type}

    def take(doc_type=None):
        """Next unused document, of a given type when one is asked for."""
        order = ([doc_type] if doc_type else []) + sorted(
            by_type, key=lambda t: cursors[t] / max(1, len(by_type[t])))
        for t in order:
            if t in by_type and cursors[t] < len(by_type[t]):
                cursors[t] += 1
                return by_type[t][cursors[t] - 1]
        return None

    bundles = []
    for _ in range(count):
        if rng.random() < singleton_rate:
            first = take()
            if first is None:
                break
            bundles.append([first])
            continue
        members, length = [], rng.randint(2, 4)
        for _slot in range(length):
            # Half the joins repeat the previous type on purpose: that is the join a
            # per-page classifier cannot see, and a corpus that leaves it to chance
            # would report a splitter as working when it only ever saw easy joins.
            want = (members[-1]["doc_type"]
                    if members and rng.random() < same_type_rate else None)
            picked = take(want) or take()
            if picked is None:
                break
            members.append(picked)
        if len(members) >= 2:
            bundles.append(members)
    return bundles


def write(root: str, bundles, out_dir: str):
    """Render each plan into one PDF, and record where its documents begin."""
    import pymupdf

    target = os.path.join(root, out_dir)
    os.makedirs(target, exist_ok=True)
    labels = []
    for index, members in enumerate(bundles):
        merged = pymupdf.open()
        spans, page = [], 0
        for member in members:
            part = open_pdf(os.path.join(root, member["file"]))
            merged.insert_pdf(part)
            spans.append({"source": member["file"], "doc_type": member["doc_type"],
                          "first_page": page, "last_page": page + part.page_count - 1})
            page += part.page_count
            part.close()
        name = f"bundle_{index:04d}.pdf"
        merged.save(os.path.join(target, name))
        merged.close()
        labels.append({
            "file": f"{out_dir}/{name}",
            "pages": page,
            "documents": spans,
            # The pages a new document starts on, page 0 excluded: it is a boundary by
            # definition and scoring it would hand every splitter a free point.
            "boundaries": [s["first_page"] for s in spans[1:]],
        })
    path = os.path.join(root, "labels", "bundles.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(labels, handle, indent=1)
    return labels, path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bundle")
    ap.add_argument("--out", default=OUT, help="dataset root")
    ap.add_argument("--into", default="bundles", help="subdirectory for the PDFs")
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--same-type-rate", type=float, default=0.5,
                    help="how often a document follows one of its own type")
    ap.add_argument("--singleton-rate", type=float, default=0.15,
                    help="how often a bundle holds exactly one document")
    a = ap.parse_args(argv)

    pool = catalogue(a.out)
    if not pool:
        raise SystemExit(f"no source documents under {a.out}; generate the corpus first")
    bundles = plan(pool, a.count, a.seed, a.same_type_rate, a.singleton_rate)
    labels, path = write(a.out, bundles, a.into)

    joins = sum(len(b["boundaries"]) for b in labels)
    same = 0
    for b in labels:
        docs = b["documents"]
        same += sum(1 for i in range(1, len(docs))
                    if docs[i]["doc_type"] == docs[i - 1]["doc_type"])
    singles = sum(1 for b in labels if len(b["documents"]) == 1)
    pages = sum(b["pages"] for b in labels)
    print(f"bundles={len(labels)}  documents={sum(len(b['documents']) for b in labels)}"
          f"  pages={pages}")
    print(f"  boundaries={joins}  of which same-type={same} "
          f"({same / joins:.0%})" if joins else "  boundaries=0")
    print(f"  single-document files={singles}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
