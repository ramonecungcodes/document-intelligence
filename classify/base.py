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


LABEL_SEPARATOR = ":"


def labels() -> tuple:
    """Every answer a classifier may give, from the type registry itself.

    Not five, nine. `form` is one document type with five variants whose field sets
    differ by more than a name -- onboarding asks for 22 fields, w4 for 9 -- and the
    extractor selects between them. A classifier that stops at `form` has answered
    half the question and left the corpus to supply the rest, which is exactly the
    hand-over Phase 3 exists to remove.
    """
    from core import doctypes
    out = []
    for name, doctype in sorted(doctypes.REGISTRY.items()):
        if doctype.variants:
            out += [f"{name}{LABEL_SEPARATOR}{v}" for v in sorted(doctype.variants)]
        else:
            out.append(name)
    return tuple(out)


def split_label(label: str):
    """`form:w9` -> ("form", "w9"); `invoice` -> ("invoice", "")."""
    head, _, tail = label.partition(LABEL_SEPARATOR)
    return head, tail


@dataclass
class Classification:
    doc_type: str                        # "" when the classifier declines to guess
    variant: str = ""                    # which field set, for a type that has several
    confidence: Optional[float] = None   # 0-1, when the classifier can say
    margin: Optional[float] = None       # how far ahead of the runner-up. A model at
                                         # 0.95 with nothing near it is not in the same
                                         # state as one at 0.95 with 0.94 behind it,
                                         # and the second probability is kept nowhere
                                         # else, so this cannot be recovered later.
    runner_up: str = ""                  # the second-best type, if there was one
    withheld: str = ""                   # what it would have said, when it abstained.
                                         # Abstaining blanks `doc_type`, and without
                                         # this the answer that was suppressed is gone
                                         # -- so a coverage curve could only ever be
                                         # drawn above whatever floor was in force, and
                                         # the question of whether that floor is set
                                         # right becomes unaskable from the artifacts.
    evidence: str = ""                   # what it matched or cited
    engine: str = ""
    seconds: float = 0.0

    @property
    def abstained(self) -> bool:
        return not self.doc_type

    @property
    def label(self) -> str:
        """Type and variant as one string, the form a model is trained on."""
        if not self.doc_type:
            return ""
        return (f"{self.doc_type}{LABEL_SEPARATOR}{self.variant}" if self.variant
                else self.doc_type)

    def provenance(self) -> dict:
        return {
            "engine": self.engine,
            "variant": self.variant or None,
            "confidence": self.confidence,
            "runner_up": self.runner_up or None,
            "withheld": self.withheld or None,
            "margin": self.margin,
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
    built = cls(**settings)
    # A composite has to build its own members, and they are configured in the same
    # manifest. Handing it the config here keeps that in one place rather than having
    # it reload and possibly disagree about which file it read.
    if hasattr(built, "bind"):
        built.bind(config)
    return built
