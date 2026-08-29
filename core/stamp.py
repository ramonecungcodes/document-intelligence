"""What produced a result, stamped into the result.

Twice in one phase a bug invalidated a set of numbers, and both times the expensive part
was not fixing it — it was working out which artifacts on disk had inherited it. The
answer came from reading commit timestamps against file mtimes, which is archaeology,
and archaeology is what you do when the record is missing.

So every report carries this, and the question "which results are still comparable"
becomes a query rather than an investigation.

FOUR THINGS MOVE INDEPENDENTLY, AND A HASH OF ONE DOES NOT COVER THE OTHERS

*The code.* A commit, plus whether the tree was dirty. Necessary and nowhere near
sufficient: a model checkpoint, a temperature, a prompt served from elsewhere and a
cache built by another engine build can all move a number without touching the
repository.

*The evaluation's meaning.* `EVALUATION` is bumped by hand, because a commit hash says
the code differed and does not say whether the difference *meant* anything. Automating
it would defeat it — a machine cannot tell a typo fix from a redefinition.

*The corpus, in two separate senses.* The labels answer "did the expected answers
change". The document bytes answer "did the inputs the system saw change". These are
different questions and conflating them loses a real signal: a corpus regenerated from
the same seed can produce different pixels — rendering, compression, DPI, degradation
artifacts — with identical labels, and extraction results can legitimately move because
of it. An earlier version of this file hashed labels only, on the reasoning that PDF
bytes churn on every rebuild. That avoided a false alarm by discarding a true one.

    same labels, same inputs        genuinely reproducible
    same labels, different inputs   same semantic corpus, different physical one
    different labels                the evaluation target itself changed

*The cohort.* Which documents were actually evaluated. Two engines can only be compared
over the same set, and this project has three different "75-document sets" in
circulation — `data/sample75.txt`, the set `reports/doctr75.jsonl` was run on, and what
`--limit 15` selects today. They overlap each other on 1 and 52 documents respectively.
A comparison across two of them would have looked entirely normal.
"""
from __future__ import annotations

import hashlib
import os
import subprocess

# Bumped by hand when the meaning of a number changes. History, newest first:
#
#   phase6-v2   repair.cli._merge now collapses optional fields. Before this, every
#               repaired record kept the {"status":..., "value":...} shape for optional
#               fields, and the scorer read it as a fabricated value. Any repair result
#               stamped earlier -- or unstamped -- overstates damage and understates net
#               delta. It moved the degraded guided arm from -0.029 to +0.002.
#   phase6-v1   first repair scorer.
EVALUATION = "phase6-v2"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = os.path.join(ROOT, "tools", "document-generator")


def _git(*args):
    try:
        out = subprocess.run(("git",) + args, cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _digest(chunks) -> str:
    out = hashlib.sha256()
    for chunk in chunks:
        out.update(chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8"))
        out.update(b"\x00")
    return out.hexdigest()[:16]


def label_fingerprint(corpus_root: str):
    """A hash of the label files: did the expected answers change.

    report.json overstated end-to-end field accuracy by three and a half points for
    weeks because the corpus was rebuilt underneath it. No commit hash could have caught
    that -- the code was correct and the labels had moved.
    """
    labels = os.path.join(corpus_root, "labels")
    if not os.path.isdir(labels):
        return None
    names = sorted(n for n in os.listdir(labels) if n.endswith(".json"))
    chunks = []
    for name in names:
        with open(os.path.join(labels, name), "rb") as handle:
            chunks.extend((name, handle.read()))
    return _digest(chunks) if chunks else None


def cohort_fingerprint(corpus_root: str, files) -> dict:
    """Which documents were evaluated, and what bytes they held.

    Two hashes, because they answer different questions. `document_set` is over the
    sorted identifiers, so it is order-independent and answers "was this the same
    cohort". `input` is over the document bytes and answers "did the system see the
    same pixels". A set hash matching with an input hash differing is a regenerated
    corpus, which is worth knowing and is not the same as an evaluation target moving.

    Only the cohort is hashed, never the whole corpus. Hashing a thousand documents on
    every report would be paid constantly to answer a question nobody asked; the
    documents that matter are the ones the run actually read.
    """
    names = sorted({str(f).replace("\\", "/") for f in (files or []) if f})
    if not names:
        return {}
    chunks, missing, total = [], 0, 0
    for name in names:
        path = os.path.join(corpus_root, name)
        if not os.path.exists(path):
            missing += 1
            continue
        with open(path, "rb") as handle:
            payload = handle.read()
        chunks.extend((name, payload))
        total += len(payload)
    return {
        "document_count": len(names),
        "document_set": _digest(names),
        "input": _digest(chunks) if chunks else None,
        "input_bytes": total,
        # Named but absent. A cohort that could not be read in full has not been
        # evaluated in full, and the count above would otherwise be a promise the
        # input hash quietly breaks.
        "documents_missing": missing or None,
    }


def generator_fingerprint() -> dict:
    """What built the corpus.

    The seed is the field that would matter most and it is not recoverable: the
    generator takes `--seed` as an argument (default 42) and writes nothing about it
    into the corpus, so a rebuilt set cannot say which seed made it. Recorded here as
    a known gap rather than omitted, because an absent field reads as "not applicable"
    and this one is "not captured".
    """
    if not os.path.isdir(GENERATOR):
        return {}
    sources = sorted(n for n in os.listdir(GENERATOR) if n.endswith(".py"))
    chunks = []
    for name in sources:
        with open(os.path.join(GENERATOR, name), "rb") as handle:
            chunks.extend((name, handle.read()))
    return {
        "generator_sources": _digest(chunks) if chunks else None,
        "generator_commit": _git("log", "-1", "--format=%h", "--",
                                 "tools/document-generator") or None,
        "seed": None,          # not recorded by the generator; see the docstring
    }


def pipeline_fingerprint(config=None, reader: str = "", extractor: str = "",
                         doctypes=()) -> dict:
    """What the pipeline was, beyond the commit.

    A model checkpoint, a temperature, a prompt served from a different build and a
    cache produced by another engine can each move a result without changing a line in
    this repository. The prompt and schema are hashed from the same helpers the runner
    uses, so a change to either shows up here rather than being inferred from a diff.
    """
    out = {"reader": reader or None, "extractor": extractor or None}
    if config is not None:
        try:
            from extract import backends
            settings = config.settings("extractor",
                                       config.chosen("extractor", extractor) or "openai",
                                       backends.OpenAIBackend.SETTINGS)
            out["extractor_model"] = settings.get("model")
            out["extractor_plugin"] = config.chosen("extractor", extractor)
        except Exception:                                   # noqa: BLE001
            pass
    if doctypes:
        try:
            from extract import schema as schema_mod

            prompts, schemas = [], []
            for spec, variant in doctypes:
                prompts.append(schema_mod.instructions(spec, variant))
                schemas.append(repr(schema_mod.json_schema(spec, variant)))
            out["prompt_hash"] = _digest(sorted(prompts))
            out["schema_hash"] = _digest(sorted(schemas))
        except Exception:                                   # noqa: BLE001
            pass
    return {k: v for k, v in out.items() if v is not None}


def stamp(corpus_root: str = "", extra: dict = None, files=None,
          config=None, reader: str = "", extractor: str = "", doctypes=()) -> dict:
    """Everything needed to decide later whether a result is still comparable."""
    commit = _git("rev-parse", "HEAD")
    out = {
        "evaluation_version": EVALUATION,
        "code": {
            "commit": commit[:12] or None,
            # True means the commit does not identify the code that ran. Not an error --
            # most exploratory runs are dirty -- but not something to discover later.
            "dirty": bool(_git("status", "--porcelain")),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        },
    }
    corpus = {}
    if corpus_root:
        corpus["root"] = corpus_root
        corpus["label"] = label_fingerprint(corpus_root)
        corpus.update(generator_fingerprint())
        if files:
            corpus.update(cohort_fingerprint(corpus_root, files))
    if corpus:
        out["corpus"] = corpus
    pipeline = pipeline_fingerprint(config, reader, extractor, doctypes)
    if pipeline:
        out["pipeline"] = pipeline
    if extra:
        out.update(extra)
    return out


def describe(stamped: dict) -> str:
    """One line, for a report header."""
    if not stamped:
        return "unstamped -- provenance unknown, treat as incomparable"
    code = stamped.get("code") or {}
    corpus = stamped.get("corpus") or {}
    bits = [f"eval {stamped.get('evaluation_version', '?')}",
            f"commit {code.get('commit') or '?'}"]
    if code.get("dirty"):
        bits.append("DIRTY (commit does not identify the code)")
    if corpus.get("label"):
        bits.append(f"labels {corpus['label']}")
    if corpus.get("document_set"):
        bits.append(f"cohort {corpus['document_set']} "
                    f"({corpus.get('document_count', '?')} docs)")
    if corpus.get("documents_missing"):
        bits.append(f"{corpus['documents_missing']} DOCUMENTS MISSING")
    return " · ".join(bits)


def comparable(one: dict, two: dict) -> tuple:
    """Can these two results go in the same table? Returns (ok, why-not, note).

    Fails closed. Anything unstamped, any difference in what a number means, any
    difference in the expected answers, and any difference in the cohort is a refusal --
    because the resulting metric is still perfectly plausible, which is what makes a
    silent denominator change dangerous.

    Differing inputs under identical labels and cohort is a *note*, not a refusal: the
    same documents were regenerated. Worth surfacing, since extraction can legitimately
    move on new pixels, and not worth blocking a comparison over.
    """
    if not one or not two:
        return False, "one of them is unstamped", ""
    if one.get("evaluation_version") != two.get("evaluation_version"):
        return False, (f"evaluation {one.get('evaluation_version')} against "
                       f"{two.get('evaluation_version')} -- a number changed meaning "
                       f"between them"), ""
    left, right = one.get("corpus") or {}, two.get("corpus") or {}
    if left.get("label") != right.get("label"):
        return False, "scored against different corpus labels", ""
    if left.get("document_set") != right.get("document_set"):
        return False, (f"different cohorts -- {left.get('document_count')} documents "
                       f"against {right.get('document_count')}, and the sets do not "
                       f"match"), ""
    note = ""
    if left.get("input") and right.get("input") and left["input"] != right["input"]:
        note = ("same documents and labels, different bytes -- the corpus was "
                "regenerated, so the system saw different pixels")
    return True, "", note
