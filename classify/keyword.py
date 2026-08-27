"""The trivial baseline: what does grep get you?

Phase 0 established that a number needs an anchor. The empty extractor scores above
zero because agreeing about blank fields is free, and knowing that is what stops the
real extractor's score from being quoted with free credit baked in. Classification
needs the same treatment, more urgently: with five types and a skewed corpus, always
answering `form` scores 0.45 without reading anything.

So this is the floor. It matches printed phrases -- an invoice says "Invoice", a
purchase order says "Purchase Order" -- and any classifier that cannot beat it is not
earning its latency.

It is a baseline rather than a candidate, and the distinction matters. The phrases are
drawn from what the generator prints, which real documents from real vendors would not
oblige it by repeating. Its score here is an upper bound on how well keyword matching
would do in production, and should be read as one.

The multi-bill case is the interesting one. A multi-bill invoice says "Invoice" as
loudly as an invoice does; only the repeated per-service structure distinguishes them.
That is exactly the confusion a model should beat a baseline on, and exactly what the
confusion matrix is for.
"""
from __future__ import annotations

import re
import time

from classify.base import Classification, register
from core.plugins import Setting

# Ordered most-specific first: a multi-bill invoice matches "invoice" too, so a plain
# invoice can only be concluded once the multi-bill signals have been ruled out.
SIGNALS = (
    ("multi_bill_invoice", (
        r"\bservice\s+code\b", r"\bcost\s+cent(?:re|er)\b", r"\bmaster\s+account\b",
        r"\bservice\s+period\b", r"\bper[- ]service\b",
    )),
    ("purchase_order", (
        r"\bpurchase\s+order\b", r"\bp\.?o\.?\s*(?:number|#)\b", r"\bship\s+to\b",
        r"\bdeliver(?:y)?\s+date\b", r"\bbuyer\b",
    )),
    ("resume", (
        r"\bwork\s+(?:history|experience)\b", r"\bemployment\s+history\b",
        r"\bskills\b", r"\beducation\b", r"\bcurriculum\s+vitae\b",
    )),
    ("form", (
        r"\bform\s+w-?[49]\b", r"\bsocial\s+security\s+number\b",
        r"\bclaim\s+number\b", r"\bloan\s+amount\b", r"\bapplicant\b",
        r"\bemployee\s+onboarding\b", r"\btaxpayer\s+identification\b",
    )),
    ("invoice", (
        r"\binvoice\b", r"\binvoice\s+number\b", r"\bbill\s+to\b",
        r"\bamount\s+due\b", r"\bremit\s+to\b",
    )),
)


@register("keyword")
class Keyword:
    """Score each type by how many of its phrases appear, and take the best."""

    SETTINGS = (
        Setting("min_hits", int, default=1,
                help="phrases a type needs before it can be chosen at all"),
        Setting("abstain_on_tie", bool, default=True,
                help="return no answer when two types score equally; a coin flip "
                     "reported as a classification is worse than an honest blank"),
    )

    def __init__(self, min_hits: int = 1, abstain_on_tie: bool = True, **_):
        self.min_hits = min_hits
        self.abstain_on_tie = abstain_on_tie

    def describe(self) -> str:
        return f"keyword · {len(SIGNALS)} types · baseline"

    def classify(self, text: str, **_) -> Classification:
        started = time.time()
        body = (text or "").lower()
        scores, matched = [], {}
        for doc_type, patterns in SIGNALS:
            hits = [p for p in patterns if re.search(p, body)]
            if hits:
                matched[doc_type] = hits[0]
            scores.append((len(hits), doc_type))
        scores.sort(key=lambda s: -s[0])

        best_hits, best = scores[0]
        second_hits, second = scores[1] if len(scores) > 1 else (0, "")

        if best_hits < self.min_hits:
            return Classification(doc_type="", engine="keyword",
                                  seconds=time.time() - started)
        if self.abstain_on_tie and best_hits == second_hits:
            return Classification(doc_type="", runner_up=second, engine="keyword",
                                  evidence=f"tied {best} / {second}",
                                  seconds=time.time() - started)

        # A ratio rather than a count: three hits out of five patterns is a different
        # kind of confident from three out of twenty, and the denominator is what makes
        # the number comparable across types.
        total = len(dict(SIGNALS)[best])
        return Classification(
            doc_type=best,
            confidence=round(best_hits / total, 3),
            runner_up=second if second_hits else "",
            evidence=matched.get(best, ""),
            engine="keyword",
            seconds=time.time() - started,
        )
