"""Neural OCR, via docTR.

The expensive end of the range: a detection model finds text regions, a recognition
model reads them. It costs a PyTorch dependency and a model download, which is why it
lives in the normalizer image and not in `di-app`.

It earns that on exactly the documents Tesseract struggles with -- low-resolution
scans, uneven lighting, perspective distortion -- which is to say the degraded corpus.
Whether it earns it on *this* corpus is a measurement, not an assumption, and the whole
point of putting both behind one seam is that the comparison is a manifest edit.

docTR returns coordinates as fractions of the page, which is a better default than
either engine's pixel space. They are converted to PDF points here so that a word box
means the same thing regardless of which engine produced it or at what resolution -- a
prerequisite for grounding, and for comparing the two engines word by word.
"""
from __future__ import annotations

import time

from core.plugins import Setting
from normalize.base import Extracted, Word, register
from normalize.native import open_pdf, source_dpi


@register("doctr")
class DocTR:
    """Detect text regions with one model, read them with another."""

    SETTINGS = (
        Setting("dpi", int, default=0,
                help="rasterisation resolution fed to the detector; 0 matches the "
                     "scan's own resolution rather than upsampling it"),
        Setting("detector", str, default="db_resnet50", help="text detection architecture"),
        Setting("recognizer", str, default="crnn_vgg16_bn", help="text recognition architecture"),
        Setting("pretrained", bool, default=True, help="download pretrained weights"),
        Setting("page_markers", bool, default=True,
                help="mark page boundaries in the text, as the native reader does"),
    )

    def __init__(self, dpi: int = 0, detector: str = "db_resnet50",
                 recognizer: str = "crnn_vgg16_bn", pretrained: bool = True,
                 page_markers: bool = True, **_):
        self.dpi = dpi
        self.detector = detector
        self.recognizer = recognizer
        self.pretrained = pretrained
        self.page_markers = page_markers
        self._predictor = None

    def describe(self) -> str:
        resolution = "source dpi" if not self.dpi else f"{self.dpi}dpi"
        return f"doctr · {self.detector} + {self.recognizer} · {resolution}"

    def predictor(self):
        """Built once and reused. Loading the weights per document is most of the cost."""
        if self._predictor is None:
            try:
                from doctr.models import ocr_predictor
            except ImportError:  # pragma: no cover - exercised only without the dependency
                raise SystemExit(
                    "the doctr normalizer needs python-doctr.\n"
                    "  pip install 'python-doctr[torch]'\n"
                    "It is installed in the di-normalizer image, not di-app.")
            self._predictor = ocr_predictor(
                det_arch=self.detector, reco_arch=self.recognizer,
                pretrained=self.pretrained)
        return self._predictor

    def read(self, path: str) -> Extracted:
        import numpy
        from PIL import Image
        import io

        started = time.time()
        document = open_pdf(path)
        try:
            images, sizes = [], []
            for page in document:
                pixmap = page.get_pixmap(dpi=self.dpi or source_dpi(page))
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                images.append(numpy.asarray(image))
                # The page's own size in points, so relative boxes can be restored to
                # the same coordinate space Tesseract and the native reader use.
                sizes.append((page.rect.width, page.rect.height))
            pages = document.page_count
        finally:
            document.close()

        if not images:
            return Extracted(text="", pages=0, layer="none", engine="doctr",
                             seconds=time.time() - started)

        result = self.predictor()(images)

        chunks, words, confidences = [], [], []
        for index, page in enumerate(result.pages, 1):
            width, height = sizes[index - 1]
            lines = []
            for block in page.blocks:
                for line in block.lines:
                    tokens = []
                    for word in line.words:
                        text = (word.value or "").strip()
                        if not text:
                            continue
                        tokens.append(text)
                        confidences.append(float(word.confidence))
                        (rx0, ry0), (rx1, ry1) = word.geometry
                        words.append(Word(
                            text=text, page=index,
                            x0=rx0 * width, y0=ry0 * height,
                            x1=rx1 * width, y1=ry1 * height,
                            confidence=float(word.confidence),
                        ))
                    if tokens:
                        lines.append(" ".join(tokens))
            body = "\n".join(lines)
            if body:
                chunks.append(f"--- page {index} ---\n{body}" if self.page_markers else body)

        text = "\n\n".join(chunks)
        return Extracted(
            text=text,
            pages=pages,
            layer="ocr" if text.strip() else "none",
            engine="doctr",
            confidence=(sum(confidences) / len(confidences)) if confidences else None,
            words=words,
            seconds=time.time() - started,
        )
