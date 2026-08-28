"""What each stage knew about a document while it was deciding, kept for later.

Phase 5 asks whether the pipeline's confidence is real. That question cannot be
answered from the predictions alone, because the predictions record what was decided
and not how sure anything was while deciding it.

Some of that is recoverable afterwards and some is not, and the difference is the whole
design of this module.

Irrecoverable, so it is written the moment it is produced:

    the classifier's confidence, its runner-up, and the margin between them
    whether a cascade escalated, and which rule made it
    whether an extraction was truncated or retried

The first of those was being thrown away. `predict_types` used the predicted type and
discarded the `Classification` that carried it, so for every document that did not
abstain, the single strongest confidence signal in the pipeline evaporated at the
moment it existed. Recovering it later means a GPU and a re-run.

Recoverable, so it is not written:

    how many fields came back empty        -- the prediction record says
    what the validators found              -- the rules are deterministic; re-run them
    how accurate the extraction was        -- the scorer and the labels say
    which degradation profile this is      -- the filename says

Storing a derived value is how two copies of one fact drift apart. The docTR cache
taught that expensively: 1,056 entries keyed to documents that had been regenerated
underneath them, every one silently stale and none of them wrong-looking.

One JSONL line per document, keyed on the corpus-relative path like every other
artifact here, written beside the predictions rather than inside them. A prediction
file is the extractor's answer and nothing else; a second stage writing into it makes
the two impossible to diff apart. And `format` on every line, so a stale signals file
raises instead of being read as current.
"""
from __future__ import annotations

import json
import os

FORMAT_VERSION = 1


def path_for(predictions_path: str) -> str:
    """The sidecar that belongs to one predictions file."""
    return os.path.splitext(predictions_path)[0] + ".signals.jsonl"


def from_classification(result) -> dict:
    """What the classifier knew, including how close the decision was.

    `margin` is recorded rather than left to be recomputed because it cannot be: the
    runner-up's probability is not kept anywhere, and the gap between first and second
    is a different signal from the first alone. A model at 0.95 with nothing near it is
    not in the same state as one at 0.95 with a 0.94 behind it.
    """
    if result is None:
        return {}
    out = {
        "margin": result.margin,
        "doc_type": result.doc_type or None,
        "variant": result.variant or None,
        "confidence": result.confidence,
        "runner_up": result.runner_up or None,
        # What it would have said. A declined document is not a document with no
        # answer, it is a document whose answer was suppressed, and the two are the
        # same record unless this is written. Every question about where the floor
        # belongs is a question about these.
        "withheld": getattr(result, "withheld", "") or None,
        "abstained": result.abstained,
        "engine": result.engine or None,
        "seconds": round(result.seconds, 3) or None,
    }
    evidence = getattr(result, "evidence", "") or ""
    if "escalated" in evidence:
        # A cascade consulted its arbiter. That is a decision the pipeline made, not a
        # property of the document, so nothing downstream can reconstruct it.
        out["escalated"] = True
        out["escalation_reason"] = evidence
    return {k: v for k, v in out.items() if v is not None}


def from_normalizer(document) -> dict:
    """What reading the page cost and how well it went."""
    if document is None:
        return {}
    out = {"layer": document.layer or None,
           "engine": document.engine or None,
           "confidence": document.confidence,
           "words": len(document.words) or None,
           "agreement": document.agreement}
    return {k: v for k, v in out.items() if v is not None}


class Writer:
    """Collects one record per document and writes them beside the predictions.

    Held in memory and written once rather than appended per document: a run that dies
    halfway should leave no signals file at all, because a partial one joined against
    complete predictions produces a calibration curve fitted to whichever documents
    happened to finish first.
    """

    def __init__(self, predictions_path: str):
        self.path = path_for(predictions_path)
        self.rows = {}

    def record(self, relative_path: str, **sections) -> None:
        key = str(relative_path).replace("\\", "/")
        row = self.rows.setdefault(key, {"format": FORMAT_VERSION, "file": key})
        for name, value in sections.items():
            if value:
                row[name] = value

    def write(self) -> str:
        with open(self.path, "w", encoding="utf-8", newline="\n") as handle:
            for key in sorted(self.rows):
                handle.write(json.dumps(self.rows[key], ensure_ascii=False) + "\n")
        return self.path

    def __len__(self) -> int:
        return len(self.rows)


def read(predictions_path: str) -> dict:
    """file -> signals, for a predictions file that has a sidecar.

    Returns an empty mapping when there is none, because the runs that predate this
    stage have no signals and asking for them is not an error -- but a sidecar written
    by a different format is, since reading it as current would silently mean something
    else.
    """
    path = path_for(predictions_path)
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
                    f"{path} was written by signals format {row.get('format')}, this "
                    f"reads {FORMAT_VERSION}. Re-run rather than trusting it.")
            rows[row["file"]] = row
    return rows
