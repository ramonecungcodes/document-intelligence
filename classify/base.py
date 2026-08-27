"""What a classifier returns, and how one gets chosen.

`Classification` carries more than a label, for the same reason `Extracted` carries
more than text.

`confidence` is what routes a document to a human in Phase 5, and it must come from the
classifier rather than being invented downstream.

`runner_up` is the cheapest useful diagnostic in the whole stage. A document called an
invoice with a multi-bill invoice close behind is a different failure from one called
an invoice with a resume close behind: the first is a hard distinction, the second
means something is badly wrong. Recording it costs nothing and turns a confusion matrix
into an explanation.

`evidence` is what the classifier saw and acted on. For the keyword baseline that is
the phrase it matched; for a model it is whatever it says convinced it. Unverified for
now, but a claim that can be checked against the page later is worth more than a bare
label, and adding it after the fact would mean re-running every classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Classification:
    doc_type: str                        # "" when the classifier declines to guess
    confidence: Optional[float] = None   # 0-1, when the classifier can say
    runner_up: str = ""                  # the second-best type, if there was one
    evidence: str = ""                   # what it matched or cited
    engine: str = ""
    seconds: float = 0.0

    @property
    def abstained(self) -> bool:
        return not self.doc_type

    def provenance(self) -> dict:
        return {
            "engine": self.engine,
            "confidence": self.confidence,
            "runner_up": self.runner_up or None,
            "evidence": (self.evidence[:120] or None),
            "seconds": round(self.seconds, 2) or None,
        }


CLASSIFIERS: dict = {}


def register(name: str):
    def wrap(cls):
        if name in CLASSIFIERS:
            raise ValueError(f"duplicate classifier {name!r}")
        CLASSIFIERS[name] = cls
        return cls
    return wrap


def build(config=None, plugin: str = "", overrides=None):
    """Construct the configured classifier.

    Defaults to `keyword` -- the baseline, not the best. A stage whose default is the
    expensive option invites reporting the expensive number and quietly forgetting what
    free would have scored.
    """
    from core import config as config_mod
    from core.plugins import SettingsError

    config = config or config_mod.load()
    chosen = (config.chosen("classifier", plugin) or "keyword").strip().lower()
    if chosen not in CLASSIFIERS:
        raise SystemExit(
            f"unknown classifier {chosen!r}; available: {', '.join(sorted(CLASSIFIERS))}")
    cls = CLASSIFIERS[chosen]
    try:
        settings = config.settings("classifier", chosen, cls.SETTINGS, overrides)
    except SettingsError as error:
        raise SystemExit(f"configuration error: {error}")
    return cls(**settings)
