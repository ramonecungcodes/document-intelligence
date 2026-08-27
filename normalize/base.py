"""What every normalizer returns, and how one gets chosen.

`Extracted` is the contract between this stage and the extractor. It deliberately
carries more than the text: `layer` says how the text was obtained, and that has to
survive into the report, because "0.83 on degraded documents" means something very
different when the text came from OCR than when the scan happened to carry a hidden
text layer. A number whose provenance is unknown is not a measurement.

`confidence` and `words` are populated by normalizers that can, and left alone by those
that cannot. Word boxes are not used yet; they are the substrate for grounding, where a
field's extracted value is checked against the region it supposedly came from. Adding
them later would mean re-running every OCR pass, so the shape is here from the start
even though nothing consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    """One recognised word and where it sits, in PDF points from the top-left."""

    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: Optional[float] = None


@dataclass
class Extracted:
    text: str
    pages: int
    layer: str                          # "native" | "ocr" | "none"
    engine: str = ""                    # which normalizer produced it
    confidence: Optional[float] = None  # mean word confidence, when the engine reports one
    words: list = field(default_factory=list)
    seconds: float = 0.0
    # Set by composite normalizers. `tried` lists what each member engine produced, so
    # a cascade that escalated can say what the cheap engine actually returned rather
    # than only that it was rejected. `agreement` is how closely independent engines
    # concurred -- a confidence signal that owes nothing to any model's self-report.
    tried: list = field(default_factory=list)
    agreement: Optional[float] = None

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def provenance(self) -> dict:
        """Safe to write into a report; says how this text came to exist."""
        return {
            "layer": self.layer,
            "engine": self.engine,
            "pages": self.pages,
            "confidence": self.confidence,
            "words": len(self.words) or None,
            "seconds": round(self.seconds, 2) or None,
            "tried": self.tried or None,
            "agreement": round(self.agreement, 4) if self.agreement is not None else None,
        }


NORMALIZERS: dict = {}


def register(name: str):
    """Decorator: make a normalizer selectable by name from the manifest."""
    def wrap(cls):
        if name in NORMALIZERS:
            raise ValueError(f"duplicate normalizer {name!r}")
        NORMALIZERS[name] = cls
        return cls
    return wrap


def build(config=None, plugin: str = "", overrides=None):
    """Construct the configured normalizer.

    Defaults to `native`, which is exactly Phase 1's behaviour -- so this stage can be
    introduced without changing a single existing number. A refactor that moves the
    baseline is a refactor you cannot check.
    """
    from core import config as config_mod
    from core.plugins import SettingsError

    config = config or config_mod.load()
    chosen = (config.chosen("normalizer", plugin) or "native").strip().lower()
    if chosen not in NORMALIZERS:
        raise SystemExit(
            f"unknown normalizer {chosen!r}; available: {', '.join(sorted(NORMALIZERS))}")

    cls = NORMALIZERS[chosen]
    try:
        settings = config.settings("normalizer", chosen, cls.SETTINGS, overrides)
    except SettingsError as error:
        raise SystemExit(f"configuration error: {error}")
    return cls(**settings)
