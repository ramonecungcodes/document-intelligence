"""Two classifiers, and a rule for when the cheap one is out of its depth.

The measurements say the two candidates fail in different places, not that one is
better. On page designs it had never seen, the image model got purchase orders wrong
0.875 of the time -- 14 of 16 read as invoices -- while getting everything else close
to right. Give a model the words and that one pair resolves outright, because an
invoice and a purchase order are both a header, a ruled line-item table and a totals
block, and what separates them is a phrase printed at the top of the page.

So the image model runs first and the text is consulted only where it is needed.

That order is not a preference. The image model reads no text at all, so it costs a
page render; the text path costs an OCR pass, which is hours over a thousand degraded
documents. Putting text first would spend that on every document to fix a minority,
and would throw away the one property that makes this stage cheap. Escalating instead
pays for OCR on the documents that actually need it.

Two triggers, both drawn from what was measured rather than guessed:

    the top two are a pair known to be confusable -- invoice against purchase order
    is the one this corpus produces, and it is a configured list rather than a
    hardcoded assumption

    confidence is under the floor, which on this model is where every error lived

A document nobody can place still comes back abstained. Escalation is a second
opinion, not a rule that something must be chosen.
"""
from __future__ import annotations

import os
import time

from classify.base import CLASSIFIERS, Classification, register
from core.plugins import Setting

DEFAULT_PAIRS = "invoice|purchase_order"


def _parse_pairs(text: str) -> list:
    """`a|b, c|d` -> [{a, b}, {c, d}]."""
    out = []
    for group in text.split(","):
        members = {p.strip() for p in group.split("|") if p.strip()}
        if len(members) > 1:
            out.append(members)
    return out


@register("cascade")
class Cascade:
    """Run a cheap classifier, and consult an expensive one where it struggles."""

    # The primary reads no text, and the secondary is asked for only on the documents
    # that escalate -- so this stage normalizes those and no others. Declaring True
    # here would hand the runner an OCR bill for the whole corpus.
    NEEDS_TEXT = False

    SETTINGS = (
        Setting("primary", str, default="dit", help="the cheap classifier, run first"),
        Setting("secondary", str, default="keyword",
                help="consulted only when the primary is in known trouble"),
        Setting("escalate_below", float, default=0.9,
                help="consult the secondary when the primary is less sure than this"),
        Setting("ambiguous", str, default=DEFAULT_PAIRS,
                help="top-two pairs that always escalate, as 'a|b, c|d'"),
        Setting("abstain_below", float, default=0.0,
                help="say nothing when even the resolved answer is under this"),
        Setting("normalizer", str, default="",
                help="how the secondary gets text; blank follows the manifest. The "
                     "runner passes its own choice here, so --normalizer means the "
                     "same thing to this stage as it does to every other one."),
    )

    def __init__(self, primary: str = "dit", secondary: str = "keyword",
                 escalate_below: float = 0.9, ambiguous: str = DEFAULT_PAIRS,
                 abstain_below: float = 0.0, normalizer: str = "", **_):
        self.normalizer_name = normalizer
        self.primary_name, self.secondary_name = primary, secondary
        self.escalate_below = escalate_below
        self.pairs = _parse_pairs(ambiguous)
        self.abstain_below = abstain_below
        self._primary = self._secondary = self._normalizer = None
        self._config = None

    def describe(self) -> str:
        pairs = ", ".join("|".join(sorted(p)) for p in self.pairs) or "none"
        return (f"cascade - {self.primary_name} then {self.secondary_name} "
                f"below {self.escalate_below} or on [{pairs}]")

    def bind(self, config) -> None:
        """Take the manifest, so the members and the normalizer come from one place."""
        self._config = config

    def _load(self):
        if self._primary is not None:
            return
        from core import config as config_mod
        config = self._config or config_mod.load()
        for name in (self.primary_name, self.secondary_name):
            if name not in CLASSIFIERS:
                raise SystemExit(f"unknown classifier {name!r} in the cascade")
        if self.primary_name == "cascade" or self.secondary_name == "cascade":
            raise SystemExit("a cascade cannot contain itself")
        # Members are built without their own abstention. The cascade owns that
        # decision: a primary that abstains internally hands back no answer and no
        # runner-up, and the escalation then has nothing to arbitrate between --
        # which turns a narrow tie-break into the secondary deciding the whole
        # question, on a document it was never meant to rule on.
        primary_settings = dict(config.settings(
            "classifier", self.primary_name, CLASSIFIERS[self.primary_name].SETTINGS))
        primary_settings["abstain_below"] = 0.0
        self._primary = CLASSIFIERS[self.primary_name](**primary_settings)
        self._secondary = CLASSIFIERS[self.secondary_name](
            **config.settings("classifier", self.secondary_name,
                              CLASSIFIERS[self.secondary_name].SETTINGS))

    def _text_for(self, path: str, document, corpus: str = ""):
        """OCR this one document, if the secondary needs text and none was handed over.

        Lazy on purpose. This is the method that decides whether the cascade is cheap.
        """
        if document is not None:
            return document
        if not getattr(self._secondary, "NEEDS_TEXT", True):
            return None
        if self._normalizer is None:
            from core import config as config_mod
            from normalize.base import NORMALIZERS, build as build_normalizer
            config = self._config or config_mod.load()
            chosen = (config.chosen("normalizer", self.normalizer_name)
                      or "native").strip().lower()
            declares = {s.name for s in NORMALIZERS.get(
                chosen, type("x", (), {"SETTINGS": ()})).SETTINGS}
            overrides = {"corpus": corpus} if corpus and "corpus" in declares else None
            self._normalizer = build_normalizer(config=config,
                                                plugin=self.normalizer_name,
                                                overrides=overrides)
        return self._normalizer.read(path)

    def _should_escalate(self, result: Classification) -> str:
        if result.abstained:
            return "primary had no answer"
        top_two = {result.doc_type, result.runner_up}
        for pair in self.pairs:
            if pair <= top_two:
                return "top two are " + "|".join(sorted(pair))
        if result.confidence is not None and result.confidence < self.escalate_below:
            return f"confidence {result.confidence:.2f} under {self.escalate_below}"
        return ""

    def classify(self, text: str = "", document=None, path: str = "",
                 corpus: str = "", **_) -> Classification:
        started = time.time()
        self._load()
        first = self._primary.classify(text, document=document, path=path)
        why = self._should_escalate(first)
        if not why:
            first.engine = f"cascade:{first.engine or self.primary_name}"
            return first

        try:
            second_doc = self._text_for(path, document, corpus)
            second = self._secondary.classify(
                (second_doc.text if second_doc is not None else text),
                document=second_doc, path=path)
        except Exception as error:
            # The escalation is an improvement, not a dependency. If OCR or the second
            # model falls over, the first answer is still an answer, and losing it
            # would make the cascade worse than the plugin it wraps.
            first.evidence = f"{first.evidence} (escalation failed: {error})".strip()
            first.engine = f"cascade:{self.primary_name}:escalation-failed"
            return first

        # The secondary arbitrates, it does not decide. It is consulted because the
        # primary was weighing two specific types, and its answer is taken only if it
        # names one of them. Anything else -- including a coarser answer, which the
        # keyword baseline gives for every form because it has no notion of variants --
        # leaves the primary's answer standing. Letting it through cost four form
        # variants their field set the first time this was run.
        considered = {first.doc_type, first.runner_up}
        arbitrated = False
        if not second.abstained:
            for pair in self.pairs:
                if pair <= considered and second.doc_type in pair:
                    arbitrated = True
                    break
        resolved = second if arbitrated else first

        out = Classification(
            doc_type=resolved.doc_type,
            variant=resolved.variant or first.variant,
            confidence=first.confidence,
            runner_up=first.doc_type if arbitrated else first.runner_up,
            evidence=f"escalated ({why}); {self.secondary_name} said "
                     f"{second.doc_type or 'nothing'}"
                     f"{'' if arbitrated else ', not taken'}",
            engine=f"cascade:{self.primary_name}->{self.secondary_name}",
            seconds=time.time() - started)
        if self.abstain_below and (out.confidence or 0) < self.abstain_below:
            out.doc_type, out.variant = "", ""
        return out
