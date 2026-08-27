"""Validators: work out whether an extracted document is wrong.

    arithmetic   money that has to add up: line items, subtotals, tax, totals

Every rule is scored twice, and the order is the point. Against the corpus *labels* a
rule that fires on a clean document is simply wrong -- there is no extraction to blame
-- and that number has to be clean before the rule goes near a prediction. Only then is
the second number, against extracted output, readable as defect detection rather than
as a measurement of the extractor wearing a validator's name.
"""
from validate.base import Finding, Report, VALIDATORS, build_all, run  # noqa: F401
from validate import arithmetic  # noqa: F401  (registration side effect)

__all__ = ["Finding", "Report", "VALIDATORS", "build_all", "run"]
