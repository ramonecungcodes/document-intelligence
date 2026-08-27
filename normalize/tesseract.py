"""Classical OCR, via Tesseract.

The cheap end of the normalizer range: no GPU, no model download, a small apt package
and a few hundred milliseconds a page. It is the baseline every fancier approach has to
beat, and the one most likely to be what a customer already has installed.

Two things about the implementation matter more than the engine choice.

Rasterisation DPI is a setting rather than a constant, because it is the single biggest
lever on both accuracy and cost, and the right value is a property of the document. The
degraded corpus includes a fax profile rendered at 170 dpi; asking Tesseract to read
that at 600 dpi upscales noise and wastes time without adding information.

Word confidence is captured and averaged, because it is the only trustworthy signal
this stage produces. It is not a model's opinion of itself -- it is an engine reporting
how well each glyph matched, which is what makes it usable as an escalation trigger for
the cascade and as an independent input to calibration later.
"""
from __future__ import annotations

import time

from core.plugins import Setting
from normalize.base import Extracted, Word, register
from normalize.native import open_pdf, source_dpi


@register("tesseract")
class Tesseract:
    """Rasterise each page, then OCR it."""

    SETTINGS = (
        Setting("dpi", int, default=0,
                help="rasterisation resolution; 0 matches the scan's own resolution, "
                     "which is almost always right -- upsampling adds noise, not detail"),
        Setting("lang", str, default="eng", help="Tesseract language pack(s), e.g. eng+fra"),
        Setting("psm", int, default=3,
                help="page segmentation mode; 3 is fully automatic, 6 assumes one block"),
        Setting("min_confidence", float, default=0.0,
                help="drop words the engine scores below this (0-100); 0 keeps everything"),
        Setting("page_markers", bool, default=True,
                help="mark page boundaries in the text, as the native reader does"),
    )

    def __init__(self, dpi: int = 0, lang: str = "eng", psm: int = 3,
                 min_confidence: float = 0.0, page_markers: bool = True, **_):
        self.dpi = dpi
        self.lang = lang
        self.psm = psm
        self.min_confidence = min_confidence
        self.page_markers = page_markers

    def describe(self) -> str:
        resolution = "source dpi" if not self.dpi else f"{self.dpi}dpi"
        return f"tesseract · {resolution} · {self.lang} · psm{self.psm}"

    def _engine(self):
        try:
            import pytesseract
        except ImportError:  # pragma: no cover - exercised only without the dependency
            raise SystemExit(
                "the tesseract normalizer needs pytesseract and the tesseract binary.\n"
                "  pip install pytesseract   and   apt-get install tesseract-ocr")
        return pytesseract

    def read(self, path: str) -> Extracted:
        pytesseract = self._engine()
        from PIL import Image
        import io

        started = time.time()
        document = open_pdf(path)
        chunks, words, confidences = [], [], []
        try:
            for index, page in enumerate(document, 1):
                # Per page, not per document: a scan can mix resolutions, and the
                # coordinate rescaling below depends on the value actually used.
                dpi = self.dpi or source_dpi(page)
                pixmap = page.get_pixmap(dpi=dpi)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                data = pytesseract.image_to_data(
                    image, lang=self.lang, config=f"--psm {self.psm}",
                    output_type=pytesseract.Output.DICT)

                # Scale OCR pixel coordinates back to PDF points, so word boxes from
                # any engine at any DPI describe the same coordinate space. Without
                # this, a box is only meaningful next to the dpi that produced it.
                scale = 72.0 / dpi
                page_words = []
                for i, raw in enumerate(data["text"]):
                    text = (raw or "").strip()
                    if not text:
                        continue
                    try:
                        confidence = float(data["conf"][i])
                    except (TypeError, ValueError):
                        confidence = -1.0
                    if confidence < 0:                       # Tesseract's "no estimate"
                        confidence = 0.0
                    if confidence < self.min_confidence:
                        continue
                    confidences.append(confidence)
                    page_words.append(Word(
                        text=text, page=index,
                        x0=data["left"][i] * scale,
                        y0=data["top"][i] * scale,
                        x1=(data["left"][i] + data["width"][i]) * scale,
                        y1=(data["top"][i] + data["height"][i]) * scale,
                        confidence=confidence / 100.0,
                    ))
                words.extend(page_words)
                body = _lines(data, self.min_confidence)
                if body:
                    chunks.append(f"--- page {index} ---\n{body}" if self.page_markers
                                  else body)
            pages = document.page_count
        finally:
            document.close()

        text = "\n\n".join(chunks)
        return Extracted(
            text=text,
            pages=pages,
            layer="ocr" if text.strip() else "none",
            engine="tesseract",
            confidence=(sum(confidences) / len(confidences) / 100.0) if confidences else None,
            words=words,
            seconds=time.time() - started,
        )


def _lines(data: dict, min_confidence: float) -> str:
    """Reassemble Tesseract's word stream into lines.

    Tesseract returns words with block/paragraph/line indices, not text. Joining on
    whitespace alone would collapse a table into one run and destroy exactly the
    structure the extractor relies on to tell a line item from a total.
    """
    lines: dict = {}
    for i, raw in enumerate(data["text"]):
        text = (raw or "").strip()
        if not text:
            continue
        try:
            if float(data["conf"][i]) < min_confidence:
                continue
        except (TypeError, ValueError):
            pass
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(text)
    return "\n".join(" ".join(words) for _, words in sorted(lines.items()))
