"""Split where the document type changes, using the classifier that already exists.

This stage needs no model of its own. Phase 3 produced a per-page document-type
predictor that reads the page as an image and costs a render, so running it on every
page of a bundle and cutting where the answer changes is close to free.

What it cannot see is stated plainly because it is most of the remaining error: two
documents of the same type back to back. A three-page invoice followed by a two-page
invoice gives `invoice` on all five pages and no signal at all. The bundle corpus forces
roughly half of its joins to be same-type for exactly this reason -- a corpus that left
them to chance would report this splitter as working when it had only been shown the
joins it can handle.

The second signal is confidence rather than a second model. A first page and a
continuation page of the same document do not look alike -- the header block, the
totals, the signature line are all on one of them -- so the classifier is measurably
less sure about continuation pages. Where it is unsure on a page whose type has not
changed, that is weak evidence of a boundary, and `split_below` decides whether to act
on it. Off by default: a threshold that has not been measured on the corpus in front of
you is a guess wearing a number's clothing.
"""
from __future__ import annotations

import time

from core.plugins import Setting
from split.base import Split, register


@register("by_type")
class ByType:
    """Classify each page; cut where the answer changes."""

    SETTINGS = (
        Setting("classifier", str, default="",
                help="which classifier plugin; blank uses the manifest's choice"),
        Setting("split_below", float, default=0.0,
                help="also cut where page confidence drops under this; 0 disables"),
    )

    def __init__(self, classifier: str = "", split_below: float = 0.0, **_):
        self.classifier_name = classifier
        self.split_below = split_below
        self._classifier = None
        self._config = None

    def bind(self, config) -> None:
        self._config = config

    def describe(self) -> str:
        floor = f" - cut under {self.split_below}" if self.split_below else ""
        return f"by_type - {self.classifier_name or 'manifest'} per page{floor}"

    def _load(self):
        if self._classifier is None:
            from classify.base import build as build_classifier
            self._classifier = build_classifier(config=self._config,
                                                plugin=self.classifier_name)

    def _page_labels(self, path: str):
        """One classification per page, each rendered on its own."""
        import pymupdf
        from normalize.native import open_pdf

        self._load()
        document = open_pdf(path)
        pages = document.page_count
        labels = []
        try:
            for index in range(pages):
                # A one-page PDF in memory, so the classifier sees a page rather than a
                # document. Writing it out and reading it back would be simpler and
                # would put a thousand temporary files on disk per run.
                single = pymupdf.open()
                single.insert_pdf(document, from_page=index, to_page=index)
                data = single.tobytes()
                single.close()
                labels.append(self._classify_bytes(data))
        finally:
            document.close()
        return pages, labels

    def _classify_bytes(self, data: bytes):
        import os
        import tempfile
        handle, temp = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(data)
            return self._classifier.classify("", document=None, path=temp)
        finally:
            os.unlink(temp)

    def split(self, path: str, **_) -> Split:
        started = time.time()
        pages, labels = self._page_labels(path)
        boundaries = []
        for index in range(1, pages):
            here, before = labels[index], labels[index - 1]
            changed = here.label and before.label and here.label != before.label
            unsure = (self.split_below and here.confidence is not None
                      and here.confidence < self.split_below)
            if changed or unsure:
                boundaries.append(index)
        confidences = [l.confidence for l in labels if l.confidence is not None]
        return Split(
            boundaries=boundaries,
            pages=pages,
            doc_types=[labels[start].doc_type for start, _end in
                       Split(boundaries=boundaries, pages=pages).spans()],
            confidence=(sum(confidences) / len(confidences)) if confidences else None,
            engine=f"by_type:{self._classifier.describe() if self._classifier else ''}",
            seconds=time.time() - started)
