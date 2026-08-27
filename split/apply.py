"""Write the documents a splitter found, so the stages after it can read them.

The splitter is the only stage that changes how many things exist, and that makes it
awkward to wire: every later stage takes a path to one document, and a bundle is not
one document. Materialising the pieces is what turns a boundary list into something the
rest of the pipeline can consume without knowing splitting happened -- the same seam
the normalizer uses when it writes text to disk for the extractor to read.

Writing rather than streaming, for the same three reasons it was right there. The
pieces are inspectable when an extraction looks wrong, so "did the splitter cut this in
half" is a question you answer by opening a file rather than by reasoning. Splitting is
deterministic and cheap to repeat but not free, and an ablation across extractors
should pay for it once. And a piece on disk has a path, which is the only thing every
downstream plugin agrees about.

The manifest carries two different kinds of claim and keeps them apart. `first_page`
and `last_page` are what the splitter decided, and are available in production.
`truth_source` is which real document this piece mostly came from, matched by page
overlap, and exists only because the corpus knows -- it is written for scoring and must
never be read by a plugin.
"""
from __future__ import annotations

import json
import os


def overlap(a_first: int, a_last: int, b_first: int, b_last: int) -> int:
    return max(0, min(a_last, b_last) - max(a_first, b_first) + 1)


def match(span, documents):
    """Which true document a predicted piece mostly is.

    Overlap rather than a start-page match, because a splitter that misses a boundary
    produces one piece covering two documents and a start-page rule would score it
    against whichever happened to be first. The piece is graded against the document it
    is mostly made of, and the leftover shows up as fields it could not find.
    """
    first, last = span
    best, best_overlap = None, 0
    for document in documents:
        n = overlap(first, last, document["first_page"], document["last_page"])
        if n > best_overlap:
            best, best_overlap = document, n
    return best, best_overlap


def apply(splitter, records, corpus_root: str, out_dir: str):
    """Split every bundle and write each document it found as its own PDF."""
    import pymupdf
    from normalize.native import open_pdf

    target = os.path.join(corpus_root, out_dir)
    os.makedirs(target, exist_ok=True)
    manifest = []

    for record in records:
        source = os.path.join(corpus_root, record["file"])
        result = splitter.split(source)
        stem = os.path.splitext(os.path.basename(record["file"]))[0]
        folder = os.path.join(target, stem)
        os.makedirs(folder, exist_ok=True)
        document = open_pdf(source)
        try:
            for index, (first, last) in enumerate(result.spans()):
                piece = pymupdf.open()
                piece.insert_pdf(document, from_page=first, to_page=last)
                name = f"{stem}_doc{index:02d}.pdf"
                piece.save(os.path.join(folder, name))
                piece.close()
                truth, shared = match((first, last), record["documents"])
                manifest.append({
                    "file": f"{out_dir}/{stem}/{name}",
                    "bundle": record["file"],
                    "first_page": first,
                    "last_page": last,
                    "pages": last - first + 1,
                    "engine": result.engine,
                    # Scoring only. A plugin that reads these is grading itself.
                    "truth_source": truth["source"] if truth else None,
                    "truth_doc_type": truth["doc_type"] if truth else None,
                    "truth_overlap_pages": shared,
                    "exact": bool(truth and first == truth["first_page"]
                                  and last == truth["last_page"]),
                })
        finally:
            document.close()

    path = os.path.join(target, "manifest.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=1)
    return manifest, path


def summarise(manifest, records) -> str:
    """What the next stage is about to be handed, and how faithful it is."""
    pieces = len(manifest)
    truth_documents = sum(len(r["documents"]) for r in records)
    exact = sum(1 for m in manifest if m["exact"])
    claimed = {}
    for m in manifest:
        if m["truth_source"]:
            claimed.setdefault(m["truth_source"], 0)
            claimed[m["truth_source"]] += 1
    split_apart = sum(1 for n in claimed.values() if n > 1)
    unseen = truth_documents - len(claimed)
    out = ["", "WHAT THE EXTRACTOR WILL RECEIVE", ""]
    out.append(f"  documents in the corpus        {truth_documents:>6}")
    out.append(f"  pieces handed downstream       {pieces:>6}")
    out.append(f"  pieces matching a document     {exact:>6}   "
               f"({exact / pieces:.1%} of pieces are a whole document, exactly)")
    out.append(f"  documents cut into pieces      {split_apart:>6}   "
               "<- extracted more than once, each time incomplete")
    out.append(f"  documents never seen alone     {unseen:>6}   "
               "<- merged into a neighbour; their fields are read off a chimera")
    return "\n".join(out)
