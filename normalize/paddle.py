"""PP-OCRv5, for the degradation the current engine loses to.

The reason to try this one rather than another model generally. docTR beat Tesseract by
1.6 points on light degradation, 1.3 on fax, and **31.5 on photographs** -- the engines
are not uniformly ranked, they are ranked by what the degradation did to the page, and
Tesseract loses specifically to perspective distortion. PP-OCRv5 ships a document
pre-processor with unwarping and orientation correction, and the `photo` profile in this
corpus is generated with the largest geometric warp of the four. That is a specific,
falsifiable reason to expect movement on a specific profile rather than a hope that a
newer model is better.

Two things this file is careful about, both of which decide whether the comparison
means anything.

**Boxes come back in the same coordinate space as every other engine.** PDF points from
the top-left, per page. Paddle returns quadrilaterals rather than rectangles, so each is
reduced to its bounding box -- a lossy step, and the right one here, because every
consumer downstream (`classify/features.py` feeding LayoutLM, the splitter, the layout
signals) speaks rectangles. An engine whose boxes lived in a different space would
score differently for reasons that have nothing to do with reading the page.

**Confidence is reported but must not be used to rank engines.** Each engine's
confidence is calibrated against itself and nothing else, and this project has already
been caught by that: the cascade normalizer reported better confidence than Tesseract
and lost to docTR by 31 points through the extractor. Confidence here feeds the routing
signals, which are scored on their own; the engine comparison goes through extraction.

Installed separately from the rest of the pipeline. OCR writes a cache and nothing
downstream imports the engine that produced it, so a second interpreter for this stage
costs nothing architecturally -- which is what made trying a package with awkward
dependencies a small decision rather than a large one.
"""
from __future__ import annotations

import os
import time

from core.plugins import Setting
from normalize.base import Extracted, Word, register
from normalize.native import open_pdf, source_dpi


@register("paddle")
class Paddle:
    """PaddleOCR PP-OCRv5 over rasterised pages."""

    SETTINGS = (
        Setting("dpi", int, default=0,
                help="render resolution; 0 follows the page's own dpi, matching what "
                     "the other engines are given so the comparison is of engines "
                     "rather than of resolutions"),
        Setting("lang", str, default="en"),
        Setting("version", str, default="PP-OCRv5",
                help="pinned, not defaulted. paddleocr 3.7.0 ships PP-OCRv6 as its "
                     "default and downloads it silently -- the first smoke run here "
                     "fetched PP-OCRv6_medium_det while the report would have said "
                     "v5. A benchmark that cannot name the model it ran is not a "
                     "benchmark"),
        Setting("unwarp", bool, default=True,
                help="PP-OCRv5's document unwarping. On by default because it is the "
                     "specific reason to try this engine -- the photo profile carries "
                     "the corpus's largest geometric distortion. Turn it off to "
                     "measure what it is worth on its own"),
        Setting("orient", bool, default=True,
                help="document orientation classification"),
        Setting("device", str, default="",
                help="blank lets Paddle choose; 'cpu' or 'gpu' to force"),
        Setting("page_markers", bool, default=True),
    )

    def __init__(self, dpi: int = 0, lang: str = "en", version: str = "PP-OCRv5",
                 unwarp: bool = True, orient: bool = True, device: str = "",
                 page_markers: bool = True, **_):
        self.dpi = dpi
        self.lang = lang
        self.version = version
        self.unwarp = unwarp
        self.orient = orient
        self.device = device
        self.page_markers = page_markers
        self._engine = None

    def describe(self) -> str:
        resolution = "source dpi" if not self.dpi else f"{self.dpi}dpi"
        extras = ",".join(filter(None, ["unwarp" if self.unwarp else "",
                                        "orient" if self.orient else ""])) or "plain"
        return f"paddle · {self.version} · {extras} · {resolution}"

    def engine(self):
        """Built once and reused. Loading the weights per document is most of the cost."""
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError:  # pragma: no cover - exercised only without the dep
                raise SystemExit(
                    "the paddle normalizer needs paddleocr.\n"
                    "  pip install paddlepaddle paddleocr\n"
                    "It is installed in .venv-paddle, not the main environment -- the "
                    "OCR stage writes a cache and nothing downstream imports it.")
            options = {
                "lang": self.lang,
                # Pinned. Left unset, the library picks whatever its current default
                # is and downloads it without comment, so the number in the report
                # would describe a model nobody chose.
                "ocr_version": self.version,
                "use_doc_unwarping": self.unwarp,
                "use_doc_orientation_classify": self.orient,
                # Textline orientation is for rotated lines within a page, which this
                # corpus does not produce; leaving it on costs time for nothing.
                "use_textline_orientation": False,
            }
            if self.device:
                options["device"] = self.device
            self._engine = PaddleOCR(**options)
        return self._engine

    def provenance(self) -> dict:
        """What actually ran, for the report rather than for the reader of this file."""
        return {"engine": "paddle", "version": self.version,
                "unwarp": self.unwarp, "orient": self.orient,
                "dpi": self.dpi or "source"}

    @staticmethod
    def _bbox(polygon):
        """A quadrilateral to a rectangle, in the raster's pixel space.

        Lossy, and correct here. Paddle returns four corners so a rotated line keeps its
        angle; every consumer in this pipeline speaks rectangles, and an engine whose
        boxes meant something different from the others' would score differently for
        reasons unrelated to reading the page.
        """
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def read(self, path: str) -> Extracted:
        import io

        import numpy
        from PIL import Image

        started = time.time()
        document = open_pdf(path)
        try:
            images, sizes, scales = [], [], []
            for page in document:
                dpi = self.dpi or source_dpi(page)
                pixmap = page.get_pixmap(dpi=dpi)
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                images.append(numpy.asarray(image))
                sizes.append((page.rect.width, page.rect.height))
                # Pixels back to points. The other engines report relative geometry and
                # multiply by the page size; Paddle reports pixels, so the conversion
                # happens here instead -- same destination, different arithmetic.
                scales.append((page.rect.width / pixmap.width,
                               page.rect.height / pixmap.height))
            pages = document.page_count
        finally:
            document.close()

        if not images:
            return Extracted(text="", pages=0, layer="none", engine="paddle",
                             seconds=time.time() - started)

        engine = self.engine()
        chunks, words, confidences = [], [], []
        for index, image in enumerate(images, 1):
            scale_x, scale_y = scales[index - 1]
            result = engine.predict(image)
            lines = []
            for block in result or []:
                # PaddleOCR 3.x returns dict-like results; 2.x returned nested lists.
                # Only the 3.x shape is supported, because PP-OCRv5 is the reason this
                # exists and quietly accepting an older layout would mean measuring a
                # different model than the one named in the report.
                texts = block.get("rec_texts") if hasattr(block, "get") else None
                if texts is None:
                    raise SystemExit(
                        "unexpected PaddleOCR result shape; this plugin targets "
                        "paddleocr 3.x (PP-OCRv5). Check the installed version.")
                scores = block.get("rec_scores") or []
                polygons = block.get("rec_polys") or block.get("dt_polys") or []
                for position, text in enumerate(texts):
                    text = (text or "").strip()
                    if not text:
                        continue
                    score = (float(scores[position])
                             if position < len(scores) else None)
                    lines.append(text)
                    if score is not None:
                        confidences.append(score)
                    if position < len(polygons):
                        x0, y0, x1, y1 = self._bbox(polygons[position])
                        words.append(Word(
                            text=text, page=index,
                            x0=x0 * scale_x, y0=y0 * scale_y,
                            x1=x1 * scale_x, y1=y1 * scale_y,
                            confidence=score,
                        ))
            body = "\n".join(lines)
            if body:
                chunks.append(f"--- page {index} ---\n{body}"
                              if self.page_markers else body)

        text = "\n\n".join(chunks)
        return Extracted(
            text=text,
            pages=pages,
            layer="ocr" if text.strip() else "none",
            engine="paddle",
            confidence=(sum(confidences) / len(confidences)) if confidences else None,
            words=words,
            seconds=time.time() - started,
        )
