"""Classifiers: work out what a document *is*.

Phase 1 and Phase 2 were both handed the document type by the corpus. That was
deliberate -- mixing extraction and classification would have made a bad number
impossible to attribute to either -- but it is not how the system will ever run. A
folder of scanned paperwork does not come labelled.

    keyword     the trivial baseline: match printed phrases. No model, no cost.
    llm         ask a model, constrained to the registry's types.
    layout      LayoutLM over words, word boxes and the page image.
    dit         the page image alone -- no words, no OCR, no normalizer.
    cascade     dit first, and the text consulted only where dit is known to struggle.

The last one is why `classify` takes more than a string. A text classifier reads what
the words say; a layout one also reads where they are, and on a fax that is most of
what is left -- docTR loses 38% of the words but the ink stays where it was. The extra
arguments are optional, so the text classifiers were not touched to add it.

The baseline exists for the same reason the empty extractor does in Phase 0. Without
it, "the classifier scores 0.94" is unreadable -- five types with a skewed corpus mean
always guessing `form` scores 0.45 for free, and a model that cannot beat grep is not
worth its latency. Every classifier is reported against both.

Classification feeds the extractor's schema choice, so an error here does not merely
mislabel a document, it asks the model for fields the document does not have. That
coupling is exactly what Phase 3 exists to measure.
"""
from classify.base import Classification, CLASSIFIERS, build  # noqa: F401
from classify import cascade, dit, keyword, layout, llm  # noqa: F401  (registration side effect)

__all__ = ["Classification", "CLASSIFIERS", "build"]
