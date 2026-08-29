"""The repair stage: a second attempt at a document something complained about.

This is the first agentic pocket in the pipeline, and it arrives last on purpose.
Everything before it is what makes it measurable -- the validators say what is wrong,
the router says which documents are worth a second look, and the scorer in
`eval/repair.py` can say whether the second look helped, against the corpus rather than
against the complaints.

The contract is deliberately narrow. A repairer is handed a document that already has an
answer and a list of reasons that answer is suspect, and returns a new record or
declines. It does not choose which documents to repair -- the router did that -- and it
does not decide whether it succeeded, because a stage that grades itself is the thing
this project spends most of its effort not building.

Bounded, and the bound is a setting rather than a convention. `max_attempts` is the
whole difference between a repair stage and an agent loop that can run until the budget
is gone. Each attempt is a model call on a document that already has an answer, so the
cost is real and the ceiling has to be visible in the manifest next to it.

Every repairer must be able to give up. `Repaired(record=None)` means the original
answer stands, and that is the correct outcome whenever the second attempt is not
clearly better-formed than the first -- returning something worse is the failure mode
the scorer exists to catch, and declining is free.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Complaint:
    """Why a document was sent for repair, in a form a prompt can carry.

    Two sources, kept apart because they license different instructions. A validator
    finding names a rule the record breaks and can be quoted at the model as a fact. A
    router gate names a threshold the document crossed, which is a statement about
    suspicion rather than about the document, and must not be phrased as though the
    page said something.
    """
    source: str                          # "validator" | "router"
    code: str
    message: str = ""
    fields: tuple = ()

    def __str__(self) -> str:
        where = f" ({', '.join(self.fields)})" if self.fields else ""
        return f"{self.message or self.code}{where}"


@dataclass
class Repaired:
    """The outcome of one repair attempt."""

    record: dict = None                  # None means the original answer stands
    attempts: int = 0
    engine: str = ""
    note: str = ""
    error: str = ""
    seconds: float = 0.0

    @property
    def changed(self) -> bool:
        return self.record is not None


REPAIRERS: dict = {}


def register(name: str):
    def wrap(cls):
        if name in REPAIRERS:
            raise ValueError(f"duplicate repairer {name!r}")
        REPAIRERS[name] = cls
        return cls
    return wrap


def complaints_for(record: dict, doctype, variant: str = "", validators=None,
                   decision=None) -> list:
    """Everything known to be wrong with this record, from both instruments.

    Validators are re-run rather than read from a stored file, for the same reason
    `route.features` re-runs them: they are deterministic given the record and the
    rules, and a stored copy of a reproducible fact is one that can go stale while
    looking current. It matters more here than anywhere else -- a repair prompted with
    a complaint about a value the record no longer holds is being asked to fix
    something that is not there.
    """
    out = []
    if validators and doctype is not None:
        from validate.base import run as run_validators

        report = run_validators(validators, record, doctype, variant)
        for finding in report.findings:
            if finding.severity != "error":
                # Warnings are not repaired. Measured at lift +0.001 against extraction
                # quality in Phase 5, they carry no information about whether the
                # document is wrong, and prompting a model to act on one is asking it
                # to change an answer for no reason.
                continue
            out.append(Complaint("validator", finding.code,
                                 getattr(finding, "message", "") or finding.code,
                                 tuple(getattr(finding, "fields", ()) or ())))
    for reason in getattr(decision, "reasons", ()) or ():
        out.append(Complaint("router", reason.gate,
                             f"{reason.gate} was {reason.value:.4g}, "
                             f"{reason.direction} the {reason.threshold:.4g} threshold"))
    return out


@dataclass
class Repairer:
    """What every repairer shares: a budget, and a way to spend one unit of it."""

    max_attempts: int = 1

    # Does attempt N build on attempt N-1, or start over?
    #
    # This is the whole difference between the two arms once a budget is larger than
    # one, and it has to be declared rather than inferred. A guided repairer iterates:
    # its second attempt sees the answer its first produced and the complaints
    # recomputed against that answer, so three attempts are a conversation. A blind
    # re-run repeats: its second attempt is the identical original request, so three
    # attempts are three independent samples.
    #
    # That makes the blind curve flat by construction, and flat is the correct null.
    # Three samples are not better than one unless something selects among them, and
    # the only available selector is the validators -- picking whichever attempt
    # satisfies them is exactly the optimisation this project's scorer exists to catch.
    # So the blind arm keeps its k-th sample, honestly, and any slope in the guided
    # curve is what iteration bought.
    ITERATIVE = False

    def describe(self) -> str:
        return f"{type(self).__name__.lower()} - up to {self.max_attempts} attempt(s)"

    def attempt(self, context) -> dict:
        raise NotImplementedError

    def repair(self, context) -> Repaired:
        """Spend the budget, stopping as soon as an attempt produces a usable record.

        Stopping on the first usable answer rather than taking the best of N is
        deliberate: choosing the best would need a judge, the only judge available is
        the validators, and picking the attempt that satisfies the validators is
        precisely the optimisation the scorer was written to catch.
        """
        started = time.time()
        attempts, last_error = 0, ""
        for _ in range(max(1, self.max_attempts)):
            attempts += 1
            try:
                record = self.attempt(context)
            except Exception as error:                      # noqa: BLE001
                last_error = f"{type(error).__name__}: {error}"
                continue
            if record:
                return Repaired(record=record, attempts=attempts,
                                engine=type(self).__name__.lower(),
                                seconds=time.time() - started)
        return Repaired(record=None, attempts=attempts,
                        engine=type(self).__name__.lower(), error=last_error,
                        note="no attempt produced a usable record",
                        seconds=time.time() - started)


@dataclass
class Context:
    """Everything an attempt needs, assembled once by the runner."""

    backend: object
    doctype: object
    variant: str
    path: str
    relative_path: str
    record: dict
    text: str = ""
    complaints: list = field(default_factory=list)
    normalizer: object = None
    rule_settings: dict = field(default_factory=dict)
    # Given a fresh record, return (merged_record, complaints_against_it). Supplied by
    # the runner, because merging harness provenance and re-running validators are its
    # concerns -- a repairer that knew how to do either would be reaching into two
    # stages it has no business knowing about.
    refresh: object = None


def build(name: str, config=None, overrides=None):
    """From the manifest's `[repairers.<name>]` block, like every other stage."""
    if name not in REPAIRERS:
        known = ", ".join(sorted(REPAIRERS)) or "none registered"
        raise SystemExit(f"unknown repairer {name!r}; known: {known}")
    cls = REPAIRERS[name]
    values = {}
    if config is not None:
        values.update(config.settings("repairer", name, cls.SETTINGS))
    values.update(overrides or {})
    return cls(**values)
