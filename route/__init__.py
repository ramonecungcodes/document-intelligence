"""Confidence and routing: deciding which documents a person has to look at.

Phase 5's question is whether the pipeline's confidence is real, and the corpus has
already answered part of it in the negative. Phase 2 found Tesseract reporting 0.88
mean word confidence on photographs whose text then extracted at 0.191. Phase 4 found
that on faxes, 0.750 of documents with no defect at all are reported defective -- the
rules being correct and measuring OCR quality instead.

Both are the same shape: a signal is useful only where it is calibrated, and the place
you most want to trust it is the place least likely to deserve it. So this stage does
not ask a model how sure it is. It collects what each stage knew while deciding, and
measures which of those actually predict a correct answer.

    signals.py   what each stage knew, kept at the moment it was known

The hard part is not collecting them, it is that they are not independent. OCR
confidence, word count, classifier confidence and validator noise all collapse together
on a fax, so seven signals may be one signal with seven names -- and a confidence model
built on them could be a fax detector wearing a calibration curve. Measuring that
correlation is a deliverable of this phase rather than an afterthought.
"""
from route.signals import Writer, from_classification, from_normalizer, read  # noqa: F401

__all__ = ["Writer", "from_classification", "from_normalizer", "read"]
