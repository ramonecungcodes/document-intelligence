"""Ask a model what the document is, constrained to the types we know.

The same backend the extractor uses, so switching models is one manifest line here too
and the classification ablation costs nothing to set up.

Three decisions worth stating.

The answer is constrained to an enum of registered types, plus "unknown". A free-text
answer would need mapping back onto the registry, and that mapping would quietly become
the classifier -- fixing "Invoice (multi-service)" into `multi_bill_invoice` is a
judgement call, and one made in a helper nobody is measuring.

"unknown" is a first-class answer rather than an absence. Phase 2 established that a
required slot with no honest answer is pressure to invent one: three fields fabricated
values on every document that lacked them, and only stopped when absence became
something the schema could express. The same trap applies here, where the cost is
worse -- a wrong type does not merely mislabel the document, it selects the wrong
extraction schema and asks the model for fields that were never on the page.

Only the first part of the document is sent. A document announces what it is at the
top; sending ten pages to decide "invoice" spends tokens on a question already answered
by the letterhead. The cap is a setting because the right value is a property of the
corpus, and being able to move it is how that gets measured rather than assumed.
"""
from __future__ import annotations

import json
import time

from classify.base import Classification, register
from core.plugins import Setting

INSTRUCTIONS = """You identify what kind of business document a page is.

Answer with one of these types, and nothing else:

  invoice             a bill for goods or services, one payable total
  multi_bill_invoice  one document billing several services separately, each with its
                      own account, period and total -- the giveaway is the repetition
  purchase_order      an order placed with a supplier, before any goods have shipped
  resume              a person's work history and skills
  form                an HR, tax or application form: W-9, W-4, onboarding, claim, loan
  unknown             you genuinely cannot tell

Answer "unknown" rather than guessing. A wrong type is worse than no type: it decides
which fields are extracted next, so a misread document is then asked for fields it
never had.

Cite the words that decided it."""


@register("llm")
class LLMClassifier:
    """One constrained call per document."""

    SETTINGS = (
        Setting("max_chars", int, default=1500,
                help="how much of the document to send; a page announces its type at "
                     "the top, and the rest is spent deciding what is already decided"),
        Setting("extractor", str, default="",
                help="which model backend to use; defaults to the configured one"),
    )

    def __init__(self, max_chars: int = 1500, extractor: str = "", **_):
        self.max_chars = max_chars
        self.extractor = extractor
        self._backend = None

    def describe(self) -> str:
        return f"llm · first {self.max_chars} chars · {self.backend().describe()}"

    def backend(self):
        if self._backend is None:
            from extract import backends
            self._backend = backends.build(plugin=self.extractor)
        return self._backend

    def schema(self) -> dict:
        from core.doctypes import REGISTRY
        return {
            "type": "object",
            "properties": {
                "doc_type": {
                    "type": "string",
                    "enum": sorted(REGISTRY) + ["unknown"],
                    "description": "The document's type, or unknown.",
                },
                "evidence": {
                    "type": "string",
                    "description": "The words on the page that decided it.",
                },
            },
            "required": ["doc_type", "evidence"],
            "additionalProperties": False,
        }

    def classify(self, text: str, **_) -> Classification:
        started = time.time()
        completion = self.backend().complete(
            system=INSTRUCTIONS,
            user=f"Document text:\n\n{(text or '')[:self.max_chars]}",
            schema=self.schema(),
        )
        if completion.error:
            return Classification(doc_type="", engine="llm",
                                  evidence=f"error: {completion.error}",
                                  seconds=time.time() - started)
        try:
            answer = json.loads(completion.text)
        except json.JSONDecodeError as error:
            return Classification(doc_type="", engine="llm",
                                  evidence=f"unparseable: {error}",
                                  seconds=time.time() - started)

        doc_type = str(answer.get("doc_type") or "").strip()
        # "unknown" is an answer, and it is recorded as an abstention rather than as a
        # sixth document type -- the scorer must not learn to treat it as a class.
        if doc_type == "unknown":
            doc_type = ""
        return Classification(
            doc_type=doc_type,
            evidence=str(answer.get("evidence") or "")[:200],
            engine="llm",
            seconds=time.time() - started,
        )
