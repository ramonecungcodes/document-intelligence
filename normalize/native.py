"""The embedded text layer, and nothing else.

Phase 1's reader, moved behind the plugin seam unchanged. It stays the default and it
stays useful: when a PDF carries real text, that text is exact and free, and no OCR
engine will ever beat it. The job of the OCR plugins is the documents this one returns
nothing for.

Returning nothing is the correct answer here, not a failure. A normalizer that guessed
at an image-only page would hide the size of the gap the OCR path has to close.
"""
from __future__ import annotations

import os
import time

from core.plugins import Setting
from normalize.base import Extracted, Word, register


def open_pdf(path: str):
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - exercised only without the dependency
        try:
            import fitz as pymupdf
        except ImportError:
            raise SystemExit("this needs PyMuPDF. Install with:  pip install pymupdf")
    return pymupdf.open(path)


def source_dpi(page, default: int = 300, floor: int = 96, ceiling: int = 600) -> int:
    """The resolution the page's own image actually carries.

    Rasterising a scan at a resolution higher than it was scanned at interpolates noise
    into more pixels and adds no information. The corpus makes that concrete: the fax
    profile is rendered at 170 dpi, and OCR-ing it at 300 was upsampling a bad image
    and then asking an engine to read the result.

    Computed from the largest embedded image: its pixel width against the width in
    points of the box it is drawn into. A page with no image -- a real vector PDF --
    has no meaningful answer, so the caller's default stands.

    Bounded at both ends. A thumbnail should not drag OCR down to 40 dpi, and a
    needlessly huge scan should not cost minutes per page for detail no engine uses.
    """
    best = 0.0
    try:
        infos = page.get_image_info()
    except Exception:                       # older PyMuPDF, or an odd page
        return default
    for info in infos:
        bbox = info.get("bbox")
        pixels = info.get("width") or 0
        if not bbox or not pixels:
            continue
        points = bbox[2] - bbox[0]
        if points <= 0:
            continue
        best = max(best, pixels / (points / 72.0))
    if best <= 0:
        return default
    return int(max(floor, min(ceiling, round(best))))


@register("native")
class NativeText:
    """Read the text layer the PDF already carries."""

    SETTINGS = (
        Setting("page_markers", bool, default=True,
                help="mark page boundaries in the text; a two-page form is one document"),
        Setting("keep_words", bool, default=False,
                help="also return word boxes, for grounding a value to a page region"),
    )

    def __init__(self, page_markers: bool = True, keep_words: bool = False, **_):
        self.page_markers = page_markers
        self.keep_words = keep_words

    def describe(self) -> str:
        return "native · embedded text layer"

    def read(self, path: str) -> Extracted:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        started = time.time()
        document = open_pdf(path)
        try:
            chunks, words = [], []
            for index, page in enumerate(document, 1):
                body = page.get_text().strip()
                if body:
                    # Page markers help the model keep multi-page documents straight; a
                    # two-page form is common and its second page is not a new document.
                    chunks.append(f"--- page {index} ---\n{body}" if self.page_markers
                                  else body)
                if self.keep_words:
                    for x0, y0, x1, y1, text, *_rest in page.get_text("words"):
                        words.append(Word(text=text, page=index,
                                          x0=x0, y0=y0, x1=x1, y1=y1))
            pages = document.page_count
        finally:
            document.close()
        text = "\n\n".join(chunks)
        return Extracted(
            text=text,
            pages=pages,
            layer="native" if text.strip() else "none",
            engine="native",
            words=words,
            seconds=time.time() - started,
        )
