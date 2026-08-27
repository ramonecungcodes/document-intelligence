"""Normalizers: get readable text out of a document, however it arrives.

Phase 1 read the embedded text layer and nothing else, which is fine until a document
arrives as a photograph of a crumpled page. On the degraded corpus that reader returns
zero characters for every single document -- not a low score, no score at all, because
there is nothing to read.

Closing that gap is a stage, not a patch, and it is a stage with genuine competing
implementations:

    native      the embedded text layer, when the PDF has one. Free and exact.
    tesseract   classical OCR over rasterised pages. Cheap, no GPU.
    doctr       neural detection + recognition. Costs a PyTorch dependency.
    cascade     cheapest first, escalating only where confidence is poor.
    consensus   all of them, keeping the best and reporting whether they agreed.
    cached      text a previous run already produced, so extraction never re-OCRs.

They are plugins for the same reason the model backends are: the interesting question
is not "does OCR work" but "which approach, at what cost, on which degradation
profile", and that question is only answerable if swapping one for another is a line of
configuration rather than a fork in the code.

Every normalizer answers the same question -- given a path, produce `Extracted` -- and
declares its own settings, so `extract.cli config` can show what each accepts without
anyone reading the source.
"""
from normalize.base import Extracted, build, NORMALIZERS  # noqa: F401
# Registration side effects. tesseract and doctr import their heavy dependencies
# lazily inside read(), so importing them here costs nothing and keeps the plugin
# listable -- `extract.cli config` can show a normalizer this image cannot run, which
# is a better error than pretending it does not exist.
from normalize import native, tesseract, doctr, composite, cached  # noqa: F401,E402

__all__ = ["Extracted", "build", "NORMALIZERS"]
