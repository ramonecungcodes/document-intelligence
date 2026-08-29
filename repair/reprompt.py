"""Ask again, and say what looked wrong.

The guided arm. It is the same request as `rerun` with one thing added: the complaints
that sent the document here, quoted back to the model alongside its own previous answer.

How the complaints are phrased is the entire design, and most of the care went into what
this prompt refuses to say.

**It does not say which value is wrong.** A validator knows that subtotal plus tax does
not equal total. It does not know which of the three the model misread, and neither do
we. A prompt that guesses -- "the total is wrong" -- is an instruction to change a field
that may well have been the only correct one, and the model will comply. The prompt
states the inconsistency and names the fields involved.

**It does not ask for a smaller answer.** The blunt way to satisfy an arithmetic rule is
to return nothing, and a model told "these fields failed validation" will sometimes
blank them. So the prompt says explicitly that dropping a value is not a fix. This is
the same failure the scorer in `eval/repair.py` was built to detect; catching it in
measurement is necessary and preventing it in the prompt is cheaper.

**It does not treat a router gate as a fact about the page.** `blank_share was 0.75,
above the 0.2 threshold` means a lot of fields came back empty, which is a statement
about the answer and not about the document -- the page may genuinely not carry those
fields. Phrased as though the document contained them, it invites invention, and an
invented value is the more expensive error: a blank field gets looked at and a confident
wrong one flows downstream unchallenged.

**And it keeps the original text.** Repair re-reads the same OCR output. If the page was
unreadable the first time it is unreadable now, and a second opinion on ruined text is
not a repair. Re-reading with a better engine is a different stage, and confusing the
two would attribute an OCR gain to a prompt.
"""
from __future__ import annotations

import json

from core.plugins import Setting
from extract import schema as schema_mod
from repair.base import Repairer, register

GUIDANCE = """You previously extracted the record below from this document. Automated
checks then flagged it. Extract the document again, using the checks as places to look
more carefully -- not as instructions about which value is wrong.

Rules for this second pass:
- The checks say something is inconsistent. They do not say which field caused it.
  Re-read the document and decide that yourself.
- Do not remove a value to satisfy a check. An empty field is not a resolved problem,
  and a record with fewer values is a worse answer, not a safer one.
- Do not invent a value the document does not contain. If a field genuinely is not on
  the page, leaving it empty is correct.
- If, after re-reading, you believe your previous answer was right, return it unchanged.

What was flagged:
{complaints}

Your previous answer:
{previous}
"""


@register("reprompt")
class Reprompt(Repairer):
    """A second extraction that has been told what looked wrong."""

    # Attempt N sees attempt N-1's answer and the complaints recomputed against it.
    # That is the arm's actual claim -- that saying what is wrong helps -- and it is
    # only testable past one attempt if the conversation continues. It also means
    # damage compounds: an attempt that made the record worse hands that worse record
    # to the next one, which the budget curve will show if it happens.
    ITERATIVE = True

    SETTINGS = (
        Setting("max_attempts", int, default=1,
                help="model calls per document. Kept equal to the rerun baseline by "
                     "default, or the comparison prices two samples against one"),
        Setting("max_complaints", int, default=6,
                help="how many findings to quote. A record breaking twenty rules is a "
                     "record that was not read at all, and listing all of them turns "
                     "the prompt into noise"),
        Setting("include_previous", bool, default=True,
                help="show the model its own previous answer. Off makes this arm a "
                     "complaint-only variant, worth measuring separately"),
    )

    def __init__(self, max_attempts: int = 1, max_complaints: int = 6,
                 include_previous: bool = True, **_):
        self.max_attempts = max_attempts
        self.max_complaints = max_complaints
        self.include_previous = include_previous

    def describe(self) -> str:
        return (f"reprompt - up to {self.max_attempts} attempt(s), "
                f"{self.max_complaints} complaints"
                f"{'' if self.include_previous else ', without the previous answer'}")

    @staticmethod
    def _previous(record: dict) -> str:
        """The prior answer, with our own bookkeeping stripped.

        `_normalizer`, `_error` and the rest are provenance this pipeline attaches, not
        anything the model said. Showing them back invites the model to reason about
        the harness instead of the document, and `_error` in particular would be read
        as a claim about the page.
        """
        return json.dumps({key: value for key, value in record.items()
                           if not key.startswith("_") and key != "file"},
                          indent=1, ensure_ascii=False, default=str)

    def _prompt(self, context) -> str:
        complaints = context.complaints[:self.max_complaints]
        if not complaints:
            # Nothing to guide with. This arm has no reason to differ from rerun here,
            # and saying "checks flagged it" with no checks would be a lie in the
            # prompt that the model has to resolve somehow.
            return ""
        lines = "\n".join(f"- {complaint}" for complaint in complaints)
        extra = len(context.complaints) - len(complaints)
        if extra > 0:
            lines += f"\n- ...and {extra} more"
        return GUIDANCE.format(
            complaints=lines,
            previous=(self._previous(context.record) if self.include_previous
                      else "(withheld)"))

    def attempt(self, context) -> dict:
        guidance = self._prompt(context)
        if not guidance:
            return None
        completion = context.backend.complete(
            system=schema_mod.instructions(context.doctype, context.variant),
            user=f"Document text:\n\n{context.text}\n\n{guidance}",
            schema=schema_mod.json_schema(context.doctype, context.variant),
        )
        if completion.error:
            raise RuntimeError(completion.error)
        parsed = json.loads(completion.text)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected an object, got {type(parsed).__name__}")
        return parsed
