"""Rule implementations.

Adding one is a single file: write the function, decorate it, import it here.

    @RULES.register("my_rule", applies_to=("invoice",), help="what it fixes")
    def my_rule(record: dict) -> int:
        ...
        return number_of_things_changed

Only encode what is true by definition. A rule that corrects the model's judgement
makes the extractor look better and the evaluation meaningless.
"""
from core.rules import RULES

from extract.rules import empty_rows, labels, rollup   # noqa: F401  (registration side effect)

__all__ = ["RULES"]
