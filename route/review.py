"""What a person found when they looked, in a form a scorer can be trained on later.

Phase 7 turns human review into training data. Whether that data is worth anything is
decided here, before any of it exists, because none of these choices can be made
retroactively.

**A review is recorded per field, not per document.** A document is not right or wrong;
it is 24 fields of which 22 are right. A document-level verdict throws away which two,
and "which two" is the entire label.

**What the reviewer did is recorded, not what is true.** A reviewer working a
700-document queue approves quickly, and the noise in that correlates with how long the
queue is -- so label quality degrades exactly when volume is highest. `action` says
`confirmed` or `corrected`; it does not say `correct`. Storing the second would bake an
assumption about reviewer attention into the ground truth, and there would be no way to
measure it afterwards. `seconds` is recorded for the same reason: a document confirmed
in two seconds and one confirmed in ninety are different evidence, and only one of them
is worth training on.

**The failure reason is recorded, not just the fact of failure.** A field that was
invented, a field that was missed, and a field copied from its neighbour are three
different defects with three different fixes, and a binary label makes them one number
that says a model is 82% accurate and nothing about what to do next. It also matters
that they are opposite errors: inventing a value where none exists is expensive because
it flows downstream unchallenged, and missing one is cheap because a blank gets looked
at.

**Why the document was in the queue is recorded.** This is the one that decides whether
the dataset is usable at all -- see `selection` below and `route.cli`'s exploration
sampling. A dataset that only contains documents the current policy flagged can teach a
model to predict the current policy, and nothing else.

Written beside the queue rather than into it. The queue is the router's answer and the
outcomes are the reviewer's; a single file that two stages write is a file where neither
can be diffed against what it was.
"""
from __future__ import annotations

import hashlib
import json
import os

FORMAT_VERSION = 1

# What the reviewer did to a field. Deliberately about the action, never the truth.
CONFIRMED = "confirmed"          # left as extracted
CORRECTED = "corrected"          # replaced the value
CLEARED = "cleared"              # removed a value that should not have been there
FILLED = "filled"                # supplied a value the extractor left blank
UNSURE = "unsure"                # the reviewer could not tell either
ACTIONS = (CONFIRMED, CORRECTED, CLEARED, FILLED, UNSURE)

# Why it was wrong. Absent when the action is `confirmed`.
#
# Invented and missed are kept apart because they are opposite errors with opposite
# costs, and a taxonomy that merges them into "wrong" loses the distinction this project
# has spent several phases establishing.
WRONG_VALUE = "wrong_value"          # a value was there and the extractor misread it
INVENTED = "invented"                # the field is not on the document at all
MISSED = "missed"                    # the field is on the document and came back blank
WRONG_SECTION = "wrong_section"      # right kind of value, taken from the wrong place
CONTAMINATED = "contaminated"        # copied from a neighbouring field
UNREADABLE = "unreadable"            # nobody could read it; not the extractor's fault
AMBIGUOUS = "ambiguous"              # the document genuinely does not settle it
REASONS = (WRONG_VALUE, INVENTED, MISSED, WRONG_SECTION, CONTAMINATED, UNREADABLE,
           AMBIGUOUS)

# How the document reached a person.
BY_GATE = "gate"                 # a routing gate fired
BY_EXPLORATION = "exploration"   # sampled at random from documents that were accepted
SELECTIONS = (BY_GATE, BY_EXPLORATION)


def explore(relative_path: str, rate: float, seed: str = "di") -> bool:
    """Should this accepted document be audited anyway?

    Hashed rather than randomly drawn, so the answer is the same every time the router
    runs over the same corpus. A fresh random draw per run would send a different sample
    to review each time, which means a document could be audited twice and never
    audited, and the exploration rate could not be reasoned about across runs.

    Rate is the share of *accepted* documents sampled. It is a real cost -- reviewer
    time spent on documents the policy already trusts -- and it buys the only unbiased
    labels the system will ever have. Without it every label describes a document the
    current policy flagged, and a model fitted on that learns to reproduce the current
    policy rather than to predict correctness. That failure is invisible: the model
    scores well, on the only documents it was ever shown.
    """
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).digest()
    # First eight bytes as a fraction of the range, which is uniform enough for
    # sampling and exactly reproducible across machines and Python versions.
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < rate


class FieldReview:
    """One field, one reviewer action."""

    __slots__ = ("field", "action", "reason", "was", "now")

    def __init__(self, field: str, action: str, reason: str = "",
                 was=None, now=None):
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; known: {', '.join(ACTIONS)}")
        if reason and reason not in REASONS:
            raise ValueError(f"unknown reason {reason!r}; known: {', '.join(REASONS)}")
        if action == CONFIRMED and reason:
            # A confirmed field has no defect. Allowing a reason here would let a row
            # say "unchanged, because it was invented", which is two claims that cannot
            # both hold and would train a scorer on the contradiction.
            raise ValueError("a confirmed field cannot carry a failure reason")
        if action != CONFIRMED and not reason:
            raise ValueError(f"action {action!r} needs a reason")
        self.field, self.action, self.reason = field, action, reason
        self.was, self.now = was, now

    @property
    def correct(self) -> bool:
        """Was the extractor right about this field.

        Derived from the action rather than stored, so it cannot disagree with it. An
        `unsure` field is not correct and not incorrect -- it is unlabelled, and callers
        training on this must drop those rather than guess, which is why this is a
        property and `labelled` sits next to it.
        """
        return self.action == CONFIRMED

    @property
    def labelled(self) -> bool:
        return self.action != UNSURE

    def to_dict(self) -> dict:
        out = {"field": self.field, "action": self.action}
        if self.reason:
            out["reason"] = self.reason
        if self.was is not None:
            out["was"] = self.was
        if self.now is not None:
            out["now"] = self.now
        return out


class Review:
    """One document, as a person left it."""

    def __init__(self, file: str, selection: str = BY_GATE, reviewer: str = "",
                 seconds: float = None, fields=None, note: str = ""):
        if selection not in SELECTIONS:
            raise ValueError(f"unknown selection {selection!r}")
        self.file = str(file).replace("\\", "/")
        self.selection = selection
        self.reviewer = reviewer
        self.seconds = seconds
        self.fields = list(fields or [])
        self.note = note

    def to_dict(self) -> dict:
        out = {
            "format": FORMAT_VERSION,
            "file": self.file,
            # The one field that decides whether this row can be used for training
            # without correcting for selection.
            "selection": self.selection,
            "fields": [f.to_dict() for f in self.fields],
            "corrected": sum(1 for f in self.fields if not f.correct and f.labelled),
            "confirmed": sum(1 for f in self.fields if f.correct),
            "unsure": sum(1 for f in self.fields if not f.labelled),
        }
        if self.reviewer:
            out["reviewer"] = self.reviewer
        if self.seconds is not None:
            # Kept so rubber-stamping is measurable later. A queue reviewed at two
            # seconds a document is a queue that produced attendance, not labels.
            out["seconds"] = round(float(self.seconds), 2)
        if self.note:
            out["note"] = self.note
        return out


def path_for(queue_path: str) -> str:
    """The outcomes file that belongs to one queue."""
    stem = queue_path[:-6] if queue_path.endswith(".jsonl") else queue_path
    return stem + ".outcomes.jsonl"


def write(queue_path: str, reviews) -> str:
    path = path_for(queue_path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for review in reviews:
            handle.write(json.dumps(review.to_dict(), ensure_ascii=False) + "\n")
    return path


def read(queue_path: str) -> dict:
    """file -> outcome, for a queue that has been reviewed.

    An empty mapping when nobody has reviewed it yet, which is not an error. A file
    written by a different format version is, since reading it as current would
    silently mean something else.
    """
    path = path_for(queue_path)
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("format") != FORMAT_VERSION:
                raise ValueError(
                    f"{path} was written by review format {row.get('format')}, this "
                    f"reads {FORMAT_VERSION}. Re-review rather than trusting it.")
            rows[row["file"]] = row
    return rows


def training_rows(queue_path: str) -> list:
    """Queue entries joined to outcomes, one row per labelled field.

    This is the shape a scorer would be fitted on, and assembling it here rather than
    in a training script is deliberate: the joins and the exclusions are decisions about
    what the dataset means, and they should live next to the schema that made them
    possible.

    `unsure` fields are dropped rather than guessed. `selection` travels with every row
    so a fit can weight or stratify by it -- a dataset that has forgotten which rows
    came from exploration cannot correct for the bias, and correcting for it is the
    entire reason exploration exists.
    """
    outcomes = read(queue_path)
    if not outcomes:
        return []
    out = []
    with open(queue_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            outcome = outcomes.get(entry.get("file"))
            if not outcome:
                continue
            for field in outcome.get("fields", []):
                if field.get("action") == UNSURE:
                    continue
                out.append({
                    "file": entry["file"],
                    "doc_type": entry.get("doc_type"),
                    "profile": entry.get("profile"),
                    "field": field["field"],
                    "selection": outcome.get("selection", BY_GATE),
                    "reviewer": outcome.get("reviewer"),
                    "seconds": outcome.get("seconds"),
                    "reason": field.get("reason"),
                    # The signals exactly as they stood when the decision was made.
                    # Recomputing them later would use whatever the code does now,
                    # which is not what the router saw.
                    "signals": entry.get("signals") or {},
                    "correct": 1 if field["action"] == CONFIRMED else 0,
                })
    return out
