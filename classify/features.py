"""Turning a document into what a layout-aware model reads.

A text classifier needs a string. A layout-aware one needs three things that have to
agree with each other: the words, a box for each word in the model's own coordinate
system, and a picture of the page. Getting them from different places is how they stop
agreeing, so they are assembled here once and used by both training and inference.

Two conventions are not negotiable and are easy to get wrong.

LayoutLM boxes are integers on a 0-1000 grid, relative to the page. Word boxes in this
project are PDF points from the top-left, so the page rectangle has to come from the
same PDF the words came from -- not from a default, and not from the clean original,
whose page size may differ from its degraded rendering.

And the image is page one, so the words must be page one too. A model shown page one's
picture beside page three's words is being trained on a contradiction.

Page one only, deliberately. Document type is a property of the first page in every
type this corpus carries, the image branch can only look at one page anyway, and 512
tokens does not reach the second page of anything.
"""
from __future__ import annotations

import os

MAX_WORDS = 512          # the model's sequence limit; more would be truncated anyway
IMAGE_SIZE = 224         # the patch grid both LayoutLMv3 and DiT expect
RENDER_SCALE = 4         # rasterise this much larger, then downsample


def _clamp(value: float, ceiling: int = 1000) -> int:
    return max(0, min(ceiling, int(round(value))))


def normalize_boxes(words, width: float, height: float):
    """Word boxes on LayoutLM's 0-1000 grid, dropping anything degenerate.

    A zero-area or inverted box is not a hard error -- OCR on a bad scan produces them
    -- but it is not information either, and the model's position embedding has no
    sensible entry for it. Dropping the word and its box together keeps the two lists
    aligned, which is the invariant everything downstream assumes.
    """
    texts, boxes = [], []
    if not width or not height:
        return texts, boxes
    for word in words:
        if getattr(word, "page", 1) != 1:
            continue
        text = (word.text or "").strip()
        if not text:
            continue
        x0, x1 = sorted((word.x0 / width * 1000, word.x1 / width * 1000))
        y0, y1 = sorted((word.y0 / height * 1000, word.y1 / height * 1000))
        box = [_clamp(x0), _clamp(y0), _clamp(x1), _clamp(y1)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        texts.append(text)
        boxes.append(box)
        if len(texts) >= MAX_WORDS:
            break
    return texts, boxes


def page_one(path: str):
    """The first page's rectangle and its picture, from the document itself."""
    from normalize.native import open_pdf
    from PIL import Image

    document = open_pdf(path)
    try:
        page = document[0]
        rect = page.rect
        # Rasterised large and then downsampled, rather than rendered straight to 224.
        # Going straight there point-samples a 170 dpi bitonal fax onto a grid coarser
        # than its own strokes: rules and text rows drop out entirely depending on
        # where the grid lands, and the page arrives looking cleaner and emptier than
        # it is. Downsampling from 4x averages those strokes into grey instead, which
        # is what tells an image model that a line was there at all.
        zoom = (IMAGE_SIZE * RENDER_SCALE) / max(rect.width, rect.height)
        pixmap = page.get_pixmap(matrix=_matrix(zoom), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        # Squared deliberately: both models want a fixed square, and letterboxing
        # would spend patches on blank margin that carries nothing.
        return rect.width, rect.height, image.resize((IMAGE_SIZE, IMAGE_SIZE),
                                                     Image.LANCZOS)
    finally:
        document.close()


def _matrix(zoom: float):
    import pymupdf
    return pymupdf.Matrix(zoom, zoom)


def features(path: str, words):
    """(words, boxes, image) for one document, all agreeing about page one."""
    width, height, image = page_one(path)
    texts, boxes = normalize_boxes(words, width, height)
    return texts, boxes, image
