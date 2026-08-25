"""One document in, one structured extraction out.

Deliberately the crudest thing that can produce a number: read the text layer, send it
once with a schema, keep what comes back. No tools, no retries on content, no repair
loop, no confidence. Those are later phases, and each of them has to justify itself
against whatever this scores.

The document type is taken from the corpus rather than predicted. Phase 1 is measuring
whether a model can read fields off a document it has never seen; classification is a
separate risk with its own phase, and mixing them would make a bad number impossible
to attribute.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from core.doctypes import DocType
from extract import schema as schema_mod
from extract.text import read_pdf

MODEL = os.environ.get("DI_MODEL", "claude-opus-5")
MAX_TOKENS = 16000

# Claude Opus 5, USD per million tokens. Used only to report what a run cost.
PRICE_IN = 5.00
PRICE_OUT = 25.00


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    calls: int = 0
    seconds: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read += other.cache_read
        self.calls += other.calls
        self.seconds += other.seconds

    @property
    def usd(self) -> float:
        return (self.input_tokens * PRICE_IN + self.output_tokens * PRICE_OUT) / 1_000_000

    def to_dict(self):
        return {
            "calls": self.calls,
            "tokens_in": self.input_tokens,
            "tokens_out": self.output_tokens,
            "cache_read_tokens": self.cache_read,
            "usd": round(self.usd, 4),
            "latency_s": round(self.seconds, 1),
        }


@dataclass
class Result:
    record: dict
    usage: Usage = field(default_factory=Usage)
    error: str = ""
    skipped: str = ""


def build_client():
    """Construct the client, failing once and legibly rather than per document.

    Zero-arg construction resolves ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN, then
    a stored `ant auth login` profile. Without any of them the SDK raises at the first
    request, which in a batch means one authentication error per document instead of
    one message saying what is missing.
    """
    try:
        import anthropic
    except ImportError:
        raise SystemExit("extract needs the Anthropic SDK. Install with:  pip install anthropic")
    client = anthropic.Anthropic()
    # The SDK resolves lazily and reports nothing until the first request, so an
    # unauthenticated batch would raise once per document instead of once.
    if not (getattr(client, "api_key", None) or getattr(client, "auth_token", None)):
        raise SystemExit(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY in your shell "
            "before running; docker compose passes it through to the container."
        )
    return client


def extract_document(client, doctype: DocType, pdf_path: str, relative_path: str,
                     effort: str = "high") -> Result:
    """Read one PDF and return a prediction record shaped like a corpus label."""
    page_text = read_pdf(pdf_path)
    base = {"file": relative_path, "doc_type": doctype.name}

    if page_text.empty:
        # No text layer: the honest answer is that this extractor cannot read it.
        base["_note"] = "no text layer"
        return Result(record=base, skipped="no text layer")

    started = time.time()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema_mod.json_schema(doctype)},
            },
            system=schema_mod.instructions(doctype),
            messages=[{
                "role": "user",
                "content": f"Document text:\n\n{page_text.text}",
            }],
        )
    except Exception as error:                     # surfaced per document, never fatal
        return Result(record=base, error=f"{type(error).__name__}: {error}")

    elapsed = time.time() - started
    usage = Usage(
        input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
        cache_read=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        calls=1,
        seconds=elapsed,
    )

    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        return Result(record=base, usage=usage,
                      error=f"refusal: {getattr(detail, 'category', 'unknown')}")

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return Result(record=base, usage=usage, error=f"unparseable JSON: {error}")

    if response.stop_reason == "max_tokens":
        base["_note"] = "truncated at max_tokens"

    base.update(parsed)
    return Result(record=base, usage=usage)
