"""Score predicted extractions against the corpus ground truth."""
from __future__ import annotations

import json
import os
from typing import Iterable, Optional

from core import doctypes
from core.doctypes import DocType, Group, NON_FIELD_KEYS
from core.normalize import Comparison, compare, is_blank
from eval.report import (
    DetectionScore,
    FieldScore,
    GroupScore,
    ScoreReport,
    SliceScore,
)

# Two rows below this similarity are treated as different rows rather than a bad
# match, which keeps a hallucinated line item from being paired with a real one.
ROW_MATCH_FLOOR = 0.34
KEY_WEIGHT = 3.0


# ------------------------------------------------------------------ loading
def load_records(path: str) -> list:
    """Accept either a JSON array or JSON Lines; both turn up in practice."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_corpus(root: str, only: Optional[Iterable[str]] = None) -> dict:
    """Map label-file stem -> list of ground-truth records."""
    labels_dir = os.path.join(root, "labels")
    if not os.path.isdir(labels_dir):
        raise SystemExit(f"no labels directory at {labels_dir}")
    wanted = set(only) if only else None
    corpus = {}
    for name in sorted(os.listdir(labels_dir)):
        if not name.endswith(".json"):
            continue
        stem = name[:-5]
        if wanted and stem not in wanted:
            continue
        corpus[stem] = load_records(os.path.join(labels_dir, name))
    return corpus


# ------------------------------------------------------------------ row matching
def _row_similarity(predicted: dict, truth: dict, group: Group) -> float:
    total = matched = 0.0
    for spec in group.fields:
        if spec.name not in truth:
            continue
        weight = KEY_WEIGHT if (spec.key or spec.name in group.keys) else 1.0
        result = compare(
            spec.kind, predicted.get(spec.name), truth.get(spec.name),
            tolerance=spec.tolerance, threshold=spec.threshold,
        )
        total += weight
        matched += weight if result.match else 0.0
    return matched / total if total else 0.0


def match_rows(predicted_rows: list, truth_rows: list, group: Group) -> list:
    """Greedily pair predicted rows with truth rows by similarity.

    Key fields are weighted heavily so an account number carries the pairing, but the
    match never *requires* one -- the defective corpus deliberately removes and
    duplicates account numbers, and those rows still have to be scored.
    """
    pairs, used_pred = [], set()
    scored = []
    for t_index, truth in enumerate(truth_rows):
        for p_index, predicted in enumerate(predicted_rows):
            score = _row_similarity(predicted, truth, group)
            if score >= ROW_MATCH_FLOOR:
                scored.append((score, t_index, p_index))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_truth = set()
    for score, t_index, p_index in scored:
        if t_index in used_truth or p_index in used_pred:
            continue
        used_truth.add(t_index)
        used_pred.add(p_index)
        pairs.append((predicted_rows[p_index], truth_rows[t_index]))
    return pairs


# ------------------------------------------------------------------ accumulation
def _field_score(bucket: dict, spec) -> FieldScore:
    if spec.name not in bucket:
        bucket[spec.name] = FieldScore(name=spec.name, kind=spec.kind)
    return bucket[spec.name]


def _group_score(bucket: dict, group: Group) -> GroupScore:
    if group.name not in bucket:
        bucket[group.name] = GroupScore(name=group.name)
    return bucket[group.name]


def score_fields(predicted: dict, truth: dict, specs, bucket: dict) -> None:
    for spec in specs:
        if spec.name not in truth:
            continue          # field does not apply to this document
        result = compare(
            spec.kind, predicted.get(spec.name), truth.get(spec.name),
            tolerance=spec.tolerance, threshold=spec.threshold,
        )
        _field_score(bucket, spec).add(result)


def score_group(predicted: dict, truth: dict, group: Group, bucket: dict) -> None:
    target = _group_score(bucket, group)
    truth_rows = truth.get(group.name) or []
    predicted_rows = predicted.get(group.name) or []
    if not isinstance(truth_rows, list) or not isinstance(predicted_rows, list):
        return
    target.truth_rows += len(truth_rows)
    target.predicted_rows += len(predicted_rows)
    for predicted_row, truth_row in match_rows(predicted_rows, truth_rows, group):
        target.matched_rows += 1
        score_fields(predicted_row, truth_row, group.fields, target.fields)
        for nested in group.groups:
            score_group(predicted_row, truth_row, nested, target.groups)


def failed(predicted: dict) -> bool:
    """A stub written by a failed extraction, not an answer to grade."""
    return bool(predicted.get("_error"))


def score_document(predicted: dict, truth: dict, spec: DocType, slice_score: SliceScore) -> None:
    if failed(predicted):
        slice_score.failed += 1
        return
    slice_score.scored += 1
    # Variant types keep most of their fields on the variant, not the shared tuple --
    # grading spec.fields alone would silently score a W-9 on one field.
    score_fields(predicted, truth, spec.graded_fields(spec.variant_of(truth)),
                 slice_score.fields)
    for group in spec.groups:
        score_group(predicted, truth, group, slice_score.groups)


# ------------------------------------------------------------------ slicing
def _slice(slices: dict, name: str, dimension: str) -> SliceScore:
    key = (dimension, name)
    if key not in slices:
        slices[key] = SliceScore(name=name, dimension=dimension)
    return slices[key]


def _degradation_of(truth: dict) -> str:
    block = truth.get("degradation")
    if isinstance(block, dict) and block.get("profile"):
        return str(block["profile"])
    return "none"


# ------------------------------------------------------------------ detection
def score_detection(pairs: list) -> DetectionScore:
    """Compare predicted `irregularities` against the injected ground truth."""
    detection = DetectionScore()
    for predicted, truth in pairs:
        expected = set(truth.get("irregularities") or [])
        found = set(predicted.get("irregularities") or [])
        if not expected:
            detection.clean_docs += 1
            if found:
                detection.clean_docs_flagged += 1
        for tag in expected & found:
            detection.add(tag, "tp")
        for tag in expected - found:
            detection.add(tag, "fn")
        for tag in found - expected:
            detection.add(tag, "fp")
    return detection


# ------------------------------------------------------------------ entry point
def score(corpus_root: str, predictions: list, only=None, provenance=None) -> ScoreReport:
    corpus = load_corpus(corpus_root, only)
    by_file = {}
    for record in predictions:
        if "file" in record:
            by_file[str(record["file"]).replace("\\", "/")] = record

    report = ScoreReport(provenance=dict(provenance or {}))
    report.cost = {"tokens_in": None, "tokens_out": None, "usd": None, "latency_s": None}
    slices, detection_pairs = {}, []
    unmatched = 0
    any_detection_prediction = False

    for stem, records in corpus.items():
        spec = doctypes.for_label_file(stem)
        if spec is None:
            report.warnings.append(f"no document type registered for labels/{stem}.json")
            continue

        for truth in records:
            key = str(truth.get("file", "")).replace("\\", "/")
            predicted = by_file.get(key)

            for name, dimension in (
                (stem, "doc_type"),
                (_degradation_of(truth), "degradation"),
                (f"layout {truth['layout']}" if "layout" in truth else None, "layout"),
            ):
                if name is None:
                    continue
                bucket = _slice(slices, name, dimension)
                bucket.docs += 1
                if predicted is not None:
                    score_document(predicted, truth, spec, bucket)

            if predicted is None:
                unmatched += 1
            elif not failed(predicted):
                detection_pairs.append((predicted, truth))
                if "irregularities" in predicted:
                    any_detection_prediction = True

    if unmatched:
        report.warnings.append(
            f"{unmatched} corpus documents had no matching prediction (joined on `file`)"
        )
    errored = sum(1 for record in by_file.values() if failed(record))
    if errored:
        report.warnings.append(
            f"{errored} predictions were extraction failures and were not graded "
            f"(they are counted as `failed`, not as wrong answers)"
        )
    extra = len(by_file) - (len(detection_pairs))
    if extra > 0:
        report.warnings.append(f"{extra} predictions did not match any corpus document")

    order = {"doc_type": 0, "degradation": 1, "layout": 2}
    report.slices = [
        slices[key] for key in sorted(slices, key=lambda k: (order.get(k[0], 9), k[1]))
    ]
    if any_detection_prediction:
        report.detection = score_detection(detection_pairs)
    return report
