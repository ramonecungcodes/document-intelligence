"""What a splitter returns, and how one gets chosen.

A splitter is handed a file that may hold several documents and says where each one
begins. It is the only stage that changes how many things exist: every stage after it
is told "this is one document", and if that is wrong the classifier types a chimera and
the extractor reads fields off two pages that have nothing to do with each other.

The contract is deliberately a list of first pages rather than a list of documents.
Boundaries are what can be checked against ground truth without agreeing on anything
else -- a splitter that finds the right cut points but disagrees about the type has
done its job, and a metric that graded whole documents would hide that.

Page 0 is always a document start and is never reported. It is a boundary by
definition, and counting it would hand every splitter, including one that does
nothing, a free correct answer per file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Split:
    """Where the documents in one file begin."""

    boundaries: list = field(default_factory=list)   # first pages, page 0 excluded
    pages: int = 0
    doc_types: list = field(default_factory=list)    # per document, when known
    confidence: Optional[float] = None
    engine: str = ""
    seconds: float = 0.0

    @property
    def count(self) -> int:
        """How many documents this file was judged to hold."""
        return len(self.boundaries) + 1 if self.pages else 0

    def spans(self):
        """(first_page, last_page) for each document found."""
        starts = [0] + sorted(self.boundaries)
        ends = [s - 1 for s in starts[1:]] + [self.pages - 1]
        return list(zip(starts, ends))

    def provenance(self) -> dict:
        return {
            "engine": self.engine,
            "pages": self.pages,
            "documents": self.count,
            "boundaries": list(self.boundaries),
            "doc_types": self.doc_types or None,
            "confidence": self.confidence,
            "seconds": round(self.seconds, 2) or None,
        }


SPLITTERS: dict = {}


def register(name: str):
    def wrap(cls):
        if name in SPLITTERS:
            raise ValueError(f"duplicate splitter {name!r}")
        SPLITTERS[name] = cls
        return cls
    return wrap


def build(config=None, plugin: str = "", overrides=None):
    """Construct the configured splitter.

    Defaults to `single` -- the do-nothing option, which is exactly what the pipeline
    did before this stage existed. A stage whose default is the clever one invites
    reporting the clever number while forgetting what doing nothing would have scored.
    """
    from core import config as config_mod
    from core.plugins import SettingsError

    config = config or config_mod.load()
    chosen = (config.chosen("splitter", plugin) or "single").strip().lower()
    if chosen not in SPLITTERS:
        raise SystemExit(
            f"unknown splitter {chosen!r}; available: {', '.join(sorted(SPLITTERS))}")
    cls = SPLITTERS[chosen]
    try:
        settings = config.settings("splitter", chosen, cls.SETTINGS, overrides)
    except SettingsError as error:
        raise SystemExit(f"configuration error: {error}")
    built = cls(**settings)
    if hasattr(built, "bind"):
        built.bind(config)
    return built
