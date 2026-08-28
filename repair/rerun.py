"""Ask the same question again, and change nothing else.

This is the baseline every other repairer is measured against, and it is the most
important thing in this package.

The extractor is sampled. Asking it the same question twice produces different answers,
and some of those answers are better -- not because anything was learned, but because
the first roll was unlucky. Any repair that sends a second request inherits that gain
for free. A guided repairer that improves documents has therefore proved nothing until
it beats this, because this improves documents too.

It is what `single` and `every_page` were to the splitter, and what the empty extractor
was to Phase 1: the number that says how much of a result is the idea and how much is
the apparatus. Phase 3 needed exactly this kind of bracketing to discover that the
model-driven splitter lost to cutting everywhere.

So it deliberately does nothing clever. It does not see the complaints. It does not
change the prompt, the temperature, the schema or the text. It is the identical call,
made twice.
"""
from __future__ import annotations

import json

from core.plugins import Setting
from extract import schema as schema_mod
from repair.base import Repairer, register


@register("rerun")
class Rerun(Repairer):
    """The same request, a second time."""

    # Every attempt is the identical original request. Nothing carries between them,
    # which is what makes this a control for sampling rather than for iteration.
    ITERATIVE = False

    SETTINGS = (
        Setting("max_attempts", int, default=1,
                help="how many times to re-ask. One, normally: this arm exists to "
                     "price a single extra sample, and giving it more budget than the "
                     "arm it baselines would flatter the baseline instead"),
    )

    def __init__(self, max_attempts: int = 1, **_):
        self.max_attempts = max_attempts

    def describe(self) -> str:
        return f"rerun - the identical request, {self.max_attempts} more time(s)"

    def attempt(self, context) -> dict:
        """The same call `extract_document` makes, with nothing added.

        Kept in step with the runner by using the same `schema_mod` helpers rather than
        a copied prompt. A baseline that drifts from the thing it baselines stops being
        a baseline and becomes a second, undocumented extractor.
        """
        completion = context.backend.complete(
            system=schema_mod.instructions(context.doctype, context.variant),
            user=f"Document text:\n\n{context.text}",
            schema=schema_mod.json_schema(context.doctype, context.variant),
        )
        if completion.error:
            raise RuntimeError(completion.error)
        parsed = json.loads(completion.text)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected an object, got {type(parsed).__name__}")
        return parsed
