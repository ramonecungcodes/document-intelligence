"""Where normalized text is kept between the normalizer and the extractor.

OCR is expensive and deterministic. Extraction is cheap and the thing we vary. Running
them in one pass gets that backwards: every model comparison would re-OCR the same
1,056 documents to produce text it already produced identically last time.

So normalization writes to disk and extraction reads from it. Three things follow.

The extractor ablation stops paying for OCR. Comparing four models over the degraded
corpus costs one OCR pass, not four.

The OCR output becomes inspectable. When extraction fails on a degraded document you
can read what the normalizer actually produced instead of guessing whether the engine
or the model was at fault -- which, on the evidence of Phase 1, is usually the wrong
guess.

And each engine gets its own tree, so two engines can be compared on identical
documents without either overwriting the other.

    <root>/<engine>/<relative-path>.json

Word boxes are stored alongside the text. They are not read yet -- grounding is a later
phase -- but re-running OCR over a thousand documents to recover coordinates that were
already computed would be a self-inflicted wound.
"""
from __future__ import annotations

import json
import os

from normalize.base import Extracted, Word

FORMAT_VERSION = 1


def path_for(root: str, engine: str, relative_path: str) -> str:
    """Cache location for one document under one engine."""
    safe = relative_path.replace("\\", "/").lstrip("/")
    return os.path.join(root, engine, safe + ".json")


def write(root: str, engine: str, relative_path: str, result: Extracted) -> str:
    target = path_for(root, engine, relative_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    payload = {
        "format": FORMAT_VERSION,
        "file": relative_path,
        "text": result.text,
        "provenance": result.provenance(),
        # Flat tuples rather than dicts: a thousand documents of word boxes is the bulk
        # of this cache, and the key names would be most of the bytes.
        "words": [[w.text, w.page, round(w.x0, 2), round(w.y0, 2),
                   round(w.x1, 2), round(w.y1, 2), w.confidence]
                  for w in result.words],
    }
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return target


def read(root: str, engine: str, relative_path: str) -> Extracted:
    target = path_for(root, engine, relative_path)
    with open(target, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"{target} was written by format {payload.get('format')}, this reads "
            f"{FORMAT_VERSION}. Re-run the normalizer rather than trusting it.")
    provenance = payload.get("provenance") or {}
    return Extracted(
        text=payload.get("text", ""),
        pages=provenance.get("pages") or 0,
        layer=provenance.get("layer") or "none",
        engine=provenance.get("engine") or engine,
        confidence=provenance.get("confidence"),
        words=[Word(text=w[0], page=w[1], x0=w[2], y0=w[3], x1=w[4], y1=w[5],
                    confidence=w[6]) for w in payload.get("words") or []],
        seconds=provenance.get("seconds") or 0.0,
        tried=provenance.get("tried") or [],
        agreement=provenance.get("agreement"),
    )


def exists(root: str, engine: str, relative_path: str) -> bool:
    return os.path.exists(path_for(root, engine, relative_path))
