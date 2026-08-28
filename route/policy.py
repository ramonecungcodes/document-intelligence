"""Deciding which documents a person has to look at.

Every measurement in Phase 5 exists to make this stage possible, and each of them
constrains it:

    the classifier's floor is 0.85, measured on the design holdout, where it answers
    47.2% of documents with nothing wrong

    the classifier's confidence predicts almost nothing about whether the *extraction*
    was right -- lift +0.028 against +0.120 for the best observable signal

    so the two decisions are separate gates and not one score

That separation is the design. A document whose type was named wrong is wrong in a way
no field can be right about, because the type chose the schema. A document whose type
was right can still come back with eleven blank fields. Collapsing both into one number
would let a confident classification vouch for a hopeless extraction, and the
measurements say that is exactly what would happen.

Gates rather than a fitted model, deliberately. A logistic regression over these signals
would score better on this corpus and would be the wrong thing to ship: it would need a
train/test split of its own, it would relearn the document-type confound that already
produced one wrong conclusion here, and nobody reviewing a queue can be told *why* a
document arrived except as a weight vector. A gate says `blank_share 0.42 over 0.30`,
which a person can check and disagree with.

Thresholds live in the manifest and are not defaulted to anything clever here. Each one
is a number `eval.cli signals` and `eval.cli calibrate` produce, and putting them in the
manifest keeps the figure that governs the pipeline next to the evidence for it rather
than buried in a class.

A gate whose signal is missing does not fire. A native-read page has no OCR confidence,
and treating absent as zero would route every clean document in the corpus to a person
on the grounds that it was illegible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.plugins import Setting

ACCEPT = "accept"
REVIEW = "review"


@dataclass
class Reason:
    """Why one gate fired, in terms a person can check against the document."""

    gate: str
    value: float
    threshold: float
    direction: str                       # "above" | "below"

    def __str__(self) -> str:
        return f"{self.gate} {self.value:.4g} {self.direction} {self.threshold:.4g}"

    def to_dict(self) -> dict:
        return {"gate": self.gate, "value": self.value,
                "threshold": self.threshold, "direction": self.direction}


@dataclass
class Decision:
    action: str
    reasons: list = field(default_factory=list)

    @property
    def review(self) -> bool:
        return self.action == REVIEW

    def to_dict(self) -> dict:
        return {"action": self.action,
                "reasons": [r.to_dict() for r in self.reasons],
                "why": "; ".join(str(r) for r in self.reasons) or None}


# Each gate: the signal it reads, which side is bad, and the manifest setting holding
# its threshold. Declaring them as data rather than as a chain of ifs means the report
# can enumerate them, a test can assert every one is reachable, and adding a gate is a
# row rather than an edit to control flow.
GATES = (
    # Classification. A wrong type is not a bad field, it is the wrong schema, so this
    # gate is about a different failure from all the others.
    ("classifier_confidence", "below", "classifier_floor"),
    # Reading. Both measured to carry real lift on degraded documents.
    ("ocr_confidence", "below", "ocr_confidence_floor"),
    ("words_per_page", "below", "words_floor"),
    # The answer's shape. The strongest single signal in the corpus: rho -0.786.
    ("blank_share", "above", "blank_share_ceiling"),
    # What the rules found. Errors only -- warnings measured at lift +0.001, which is
    # nothing, and gating on them would send documents to a person for no reason.
    ("validator_errors", "above", "validator_errors_ceiling"),
)


class Policy:
    """Apply the gates to one document's signals."""

    SETTINGS = (
        Setting("classifier_floor", float, default=0.85,
                help="decline below this classifier confidence. 0.85 is the measured "
                     "zero-error floor on the design holdout; 0 disables the gate"),
        Setting("blank_share_ceiling", float, default=0.0,
                help="review when this share of expected fields came back empty. The "
                     "strongest measured signal; 0 disables"),
        Setting("ocr_confidence_floor", float, default=0.0,
                help="review when mean OCR word confidence is under this; 0 disables"),
        Setting("words_floor", float, default=0.0,
                help="review when the page yielded fewer words than this; 0 disables"),
        Setting("validator_errors_ceiling", float, default=0.0,
                help="review when more than this many rules failed at severity error. "
                     "0 means any error routes the document, which is the point of "
                     "having errors; a negative number disables the gate"),
    )

    def __init__(self, classifier_floor: float = 0.85,
                 blank_share_ceiling: float = 0.0,
                 ocr_confidence_floor: float = 0.0,
                 words_floor: float = 0.0,
                 validator_errors_ceiling: float = 0.0, **_):
        self.thresholds = {
            "classifier_floor": classifier_floor,
            "blank_share_ceiling": blank_share_ceiling,
            "ocr_confidence_floor": ocr_confidence_floor,
            "words_floor": words_floor,
            "validator_errors_ceiling": validator_errors_ceiling,
        }

    def describe(self) -> str:
        on = [f"{name}={value:g}" for name, value in sorted(self.thresholds.items())
              if self._enabled(name, value)]
        return "policy - " + (", ".join(on) if on else "every document accepted")

    @staticmethod
    def _enabled(name: str, threshold) -> bool:
        """Zero disables every gate except the validator one, where zero is the point.

        The asymmetry is real and worth stating rather than smoothing over. "Confidence
        under 0" and "fewer than 0 words" describe no document, so zero is the natural
        way to switch those off. "More than 0 errors" describes exactly the documents a
        validator exists to find, so zero has to mean on -- and a negative number is
        then the way to turn it off.
        """
        if threshold is None:
            return False
        if name == "validator_errors_ceiling":
            return threshold >= 0
        return threshold > 0

    def decide(self, signals: dict) -> Decision:
        reasons = []
        for signal, direction, setting in GATES:
            threshold = self.thresholds.get(setting)
            if not self._enabled(setting, threshold):
                continue
            value = (signals or {}).get(signal)
            # Missing is not bad. A clean page has no OCR confidence, and reading
            # absence as zero would route the whole clean corpus as illegible.
            if value is None:
                continue
            fired = value > threshold if direction == "above" else value < threshold
            if fired:
                reasons.append(Reason(signal, float(value), float(threshold),
                                      direction))
        return Decision(REVIEW if reasons else ACCEPT, reasons)


def build(config=None, overrides=None) -> Policy:
    """From the manifest's `[routers.policy]` block, like every other stage."""
    values = {}
    if config is not None:
        values.update(config.settings("router", "policy", Policy.SETTINGS))
    values.update(overrides or {})
    return Policy(**values)
