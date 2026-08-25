"""Get text out of a PDF.

Phase 1 reads the embedded text layer and nothing else. Documents that have no text
layer -- the degraded scans -- come back empty, and that is the honest result rather
than a bug to paper over: it is the measured size of the gap that OCR has to close,
and the reason the normalizer becomes its own pipeline stage in Phase 2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Extracted:
    text: str
    pages: int
    layer: str          # "native" when the PDF carries text, "none" when it does not

    @property
    def empty(self) -> bool:
        return not self.text.strip()


def _open(path: str):
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - exercised only without the dependency
        try:
            import fitz as pymupdf
        except ImportError:
            raise SystemExit("extract needs PyMuPDF. Install with:  pip install pymupdf")
    return pymupdf.open(path)


def read_pdf(path: str) -> Extracted:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    document = _open(path)
    try:
        chunks = []
        for index, page in enumerate(document, 1):
            body = page.get_text().strip()
            if body:
                # Page markers help the model keep multi-page documents straight; a
                # two-page form is common and its second page is not a new document.
                chunks.append(f"--- page {index} ---\n{body}")
        pages = document.page_count
    finally:
        document.close()
    text = "\n\n".join(chunks)
    return Extracted(text=text, pages=pages, layer="native" if text.strip() else "none")
