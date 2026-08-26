"""Post-extraction rules: deterministic cleanup, declared per document type.

A rule fixes something that is wrong *by definition* -- structurally impossible, the
kind of thing a validator would reject outright. A totals row is not a line item; a row
with no description and no amount is not a row.

What a rule must never do is correct the model's judgement. If a stated total disagrees
with its line items, that is a finding for the validators and a signal the defect
corpus exists to measure. Silently repairing it would make the extractor look better
and the system worse, and the evaluation would stop measuring the model at all.

Two properties keep that line visible:

    every rule declares which document types it applies to, so a fix for one type
    cannot quietly reshape another;

    every rule reports what it changed, so a run can say "these numbers include 4 rows
    removed by a rule" instead of absorbing the difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Rule:
    """One deterministic transformation.

    `apply` takes the prediction record and returns the number of things it changed.
    It mutates in place: records are plain dicts and copying them per rule would make
    the count meaningless.
    """

    name: str
    apply: Callable[[dict], int]
    applies_to: tuple = ()      # doctype names; empty means every type
    help: str = ""

    def wanted_for(self, doctype_name: str) -> bool:
        return not self.applies_to or doctype_name in self.applies_to


@dataclass
class RuleReport:
    """What the rules did to one document."""

    applied: dict = field(default_factory=dict)

    def record(self, name: str, changed: int) -> None:
        if changed:
            self.applied[name] = self.applied.get(name, 0) + changed

    @property
    def total(self) -> int:
        return sum(self.applied.values())

    def to_dict(self) -> dict:
        return dict(self.applied)


class Registry:
    """The available rules, and which of them are switched on."""

    def __init__(self):
        self._rules: dict = {}

    def add(self, rule: Rule) -> Rule:
        if rule.name in self._rules:
            raise ValueError(f"duplicate rule {rule.name!r}")
        self._rules[rule.name] = rule
        return rule

    def register(self, name: str, applies_to=(), help: str = ""):
        """Decorator form: @rules.register("drop_empty_rows", applies_to=("invoice",))"""
        def wrap(func):
            self.add(Rule(name=name, apply=func, applies_to=tuple(applies_to), help=help))
            return func
        return wrap

    def __contains__(self, name):
        return name in self._rules

    def __iter__(self):
        return iter(sorted(self._rules.values(), key=lambda r: r.name))

    def names(self):
        return sorted(self._rules)

    def enabled(self, settings: Optional[dict] = None):
        """Rules switched on, in name order.

        Defaults to on: a rule ships because it is correct, and turning one off is the
        deliberate act. `settings` is the manifest's [rules] block.
        """
        settings = settings or {}
        unknown = [k for k in settings if k not in self._rules]
        if unknown:
            raise ValueError(
                f"[rules] has no rule {unknown[0]!r}; available: {', '.join(self.names())}")
        return [r for r in self if settings.get(r.name, True)]

    def apply(self, record: dict, doctype_name: str,
              settings: Optional[dict] = None) -> RuleReport:
        report = RuleReport()
        for rule in self.enabled(settings):
            if rule.wanted_for(doctype_name):
                report.record(rule.name, rule.apply(record) or 0)
        return report

    def describe(self) -> str:
        lines = []
        for rule in self:
            scope = ", ".join(rule.applies_to) if rule.applies_to else "all types"
            lines.append(f"  {rule.name:<22} {scope}")
            if rule.help:
                lines.append(f"  {'':<22} {rule.help}")
        return "\n".join(lines) or "  (none registered)"


RULES = Registry()
