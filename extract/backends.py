"""Model backends: one interface, two implementations.

The extractor was always meant to be swappable -- that is the whole argument for the
plugin seams -- and a second backend is the first thing that actually proves it. It
also buys the ablation early: the same corpus, the same schema, a local model against
a frontier one, scored by the same harness.

    anthropic   the Claude API
    openai      anything speaking the OpenAI chat-completions API: LM Studio,
                Ollama, vLLM, llama.cpp, or a hosted endpoint behind a proxy

Both are configured entirely from the environment, so switching is one variable and
never a code change.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from core.plugins import Setting

DEFAULT_MAX_TOKENS = 8000
DEFAULT_TIMEOUT = 600.0

# USD per million tokens, for reporting what a run cost. Local inference is free, so
# a backend with no price simply reports zero rather than pretending to a number.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def extract_json(text: str):
    """Pull a JSON object out of a model response.

    Only needed in prompt mode. Models wrap JSON in code fences, prefix it with a
    sentence, or emit it bare; all three turn up on the same endpoint.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if "```" in body[3:] else body[3:]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
        body = body.strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    start = body.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object in response", body, 0)
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(body[start:index + 1])
    raise json.JSONDecodeError("unterminated JSON object", body, start)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    seconds: float = 0.0
    usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.calls += other.calls
        self.seconds += other.seconds
        self.usd += other.usd

    def to_dict(self):
        return {
            "calls": self.calls,
            "tokens_in": self.input_tokens,
            "tokens_out": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens or None,
            "usd": round(self.usd, 4),
            "latency_s": round(self.seconds, 1),
        }


@dataclass
class Completion:
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    error: str = ""
    truncated: bool = False
    mode: str = ""          # which JSON strategy produced this


def _price(model: str, usage: Usage) -> float:
    rates = PRICING.get(model)
    if not rates:
        return 0.0
    return (usage.input_tokens * rates[0] + usage.output_tokens * rates[1]) / 1_000_000


# ------------------------------------------------------------------ anthropic
class AnthropicBackend:
    name = "anthropic"
    SETTINGS = (
        Setting("model", str, default="claude-opus-5",
                help="Claude model id, e.g. claude-opus-5"),
        Setting("effort", str, default="high",
                help="low | medium | high | xhigh | max"),
        Setting("max_tokens", int, default=DEFAULT_MAX_TOKENS),
        Setting("api_key", str, secret=True,
                help="falls back to ANTHROPIC_API_KEY or an `ant auth login` profile"),
    )

    def __init__(self, model: str, max_tokens: int, effort: str = "high",
                 api_key: str = "", **_):
        try:
            import anthropic
        except ImportError:
            raise SystemExit("pip install anthropic")
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        if not (getattr(client, "api_key", None) or getattr(client, "auth_token", None)):
            raise SystemExit(
                "No Anthropic credentials found. Set ANTHROPIC_API_KEY, or switch "
                "backends with DI_BACKEND=openai."
            )
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def describe(self) -> str:
        return f"anthropic · {self.model} · effort {self.effort}"

    def complete(self, system: str, user: str, schema: dict) -> Completion:
        started = time.time()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as error:
            return Completion(error=f"{type(error).__name__}: {error}")

        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            calls=1,
            seconds=time.time() - started,
        )
        usage.usd = _price(self.model, usage)

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            return Completion(usage=usage,
                              error=f"refusal: {getattr(detail, 'category', 'unknown')}")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return Completion(text=text, usage=usage,
                          truncated=response.stop_reason == "max_tokens")


# ------------------------------------------------------------------ openai-compatible
class OpenAIBackend:
    """Any server speaking OpenAI chat-completions.

    Two details bite in practice and are handled here rather than left to the caller:

    Reasoning models put their chain of thought in `reasoning_content` and the answer
    in `content`. If `max_tokens` is small the reasoning consumes the whole budget and
    `content` comes back empty with `finish_reason: "length"` -- a silent failure that
    looks like the model ignoring the schema. The default budget is generous and
    truncation is reported explicitly.

    Endpoints behind a reverse proxy often use HTTP Basic auth, which is not the same
    thing as an OpenAI bearer token. DI_BASIC_AUTH sends the right header instead.

    And structured output is not universal. Some engines accept a `response_format` of
    json_schema and then return an empty completion, because constrained decoding is
    not actually implemented for that model -- the request succeeds and produces
    nothing. Auto mode notices and carries the schema in the prompt instead, which
    every model can manage.
    """

    name = "openai"
    SETTINGS = (
        # Not `required`: build() checks it separately so an unset model is answered
        # with the endpoint's actual model list rather than a generic complaint.
        Setting("model", str,
                help="a model the endpoint serves; leave unset to be shown the list"),
        Setting("base_url", str, default="http://host.docker.internal:1234/v1",
                help="OpenAI-compatible endpoint. Inside a container, localhost is the "
                     "container -- use host.docker.internal for a server on this machine"),
        Setting("api_key", str, secret=True,
                help="sent as Authorization: Bearer. Local servers ignore it"),
        Setting("basic_auth", str, secret=True,
                help="user:pass, for an endpoint behind HTTP Basic auth instead"),
        Setting("max_tokens", int, default=DEFAULT_MAX_TOKENS,
                help="reasoning models spend this thinking before they answer"),
        Setting("timeout", float, default=DEFAULT_TIMEOUT, help="seconds per request"),
        Setting("json_mode", str, default="auto",
                help="auto | schema | prompt. auto falls back to schema-in-prompt when "
                     "an endpoint accepts json_schema but returns nothing"),
        Setting("no_think", bool, default=False,
                help="disable chain-of-thought; extraction is transcription"),
    )

    def __init__(self, model: str, max_tokens: int, base_url: str,
                 api_key: str = "", basic_auth: str = "", timeout: float = DEFAULT_TIMEOUT,
                 json_mode: str = "auto", no_think: bool = False, **_):
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("pip install openai")
        headers = {}
        if basic_auth:
            token = base64.b64encode(basic_auth.encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        self.client = OpenAI(
            base_url=base_url,
            # Local servers ignore the key but the SDK requires a non-empty one.
            api_key=api_key or "not-needed",
            default_headers=headers or None,
            timeout=timeout,
            max_retries=2,
        )
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.auth = "basic" if basic_auth else ("bearer" if api_key else "none")
        # "schema" constrains decoding server-side; "prompt" carries the schema in the
        # system message and parses JSON back out of ordinary text. "auto" starts on
        # schema and drops to prompt the first time it comes back empty.
        self.json_mode = json_mode
        self._mode = "schema" if json_mode in ("auto", "schema") else "prompt"
        self._downgraded = False
        self._lock = threading.Lock()
        # Extraction is transcription, not reasoning: the fields are printed on the
        # page. Thinking tokens are latency spent on a task that does not need them,
        # and they give the model room to "correct" a document the prompt explicitly
        # says to copy verbatim. Qwen-family templates take this switch.
        self.no_think = no_think

    def describe(self) -> str:
        thinking = " · no-think" if self.no_think else ""
        return (f"openai · {self.model} · {self.base_url} · auth {self.auth}"
                f" · json {self.json_mode}{thinking}")

    def available_models(self):
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception:
            return []

    def _call(self, system: str, user: str, schema: dict, mode: str):
        kwargs = {}
        if mode == "schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": schema},
            }
        else:
            # No constrained decoding available, so the schema goes in the prompt and
            # the JSON gets parsed back out of ordinary text.
            system = (
                system
                + "\n\nReply with a single JSON object and nothing else."
                + " It must match this JSON Schema exactly:\n"
                + json.dumps(schema)
            )
        if self.no_think:
            # The chat-template switch Qwen and friends read. Servers that do not
            # recognise it ignore the field rather than failing the request.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        started = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        raw_usage = getattr(response, "usage", None)
        details = getattr(raw_usage, "completion_tokens_details", None)
        usage = Usage(
            input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
            calls=1,
            seconds=time.time() - started,
        )
        usage.usd = _price(self.model, usage)
        return response.choices[0], usage

    def _modes(self):
        """The strategies to try, resolved up front.

        This has to be a list rather than a generator: _downgrade() mutates
        self._mode mid-attempt, and a generator that re-read it would skip the very
        fallback it just decided to use. Documents also run concurrently, so the
        snapshot keeps one worker's downgrade from reshaping another's plan.
        """
        with self._lock:
            current = self._mode
        if current == "schema" and self.json_mode == "auto":
            return ["schema", "prompt"]
        return [current]

    def _downgrade(self):
        with self._lock:
            if self._downgraded:
                return
            self._downgraded = True
            self._mode = "prompt"
        print(f"  note: {self.model} returned nothing under json_schema; using "
              f"schema-in-prompt for the rest of this run", flush=True)

    def complete(self, system: str, user: str, schema: dict) -> Completion:
        total = Usage()
        for attempt, mode in enumerate(self._modes()):
            attempt_started = time.time()
            try:
                choice, usage = self._call(system, user, schema, mode)
            except Exception as error:
                # Record the wall time even on failure: a request that timed out took
                # exactly as long as the budget it blew.
                total.add(Usage(calls=1, seconds=time.time() - attempt_started))
                return Completion(usage=total, mode=mode,
                                  error=f"{type(error).__name__}: {error}")
            total.add(usage)
            text = (choice.message.content or "").strip()
            truncated = choice.finish_reason == "length"

            if text:
                if mode == "prompt":
                    try:
                        text = json.dumps(extract_json(text))
                    except json.JSONDecodeError as error:
                        return Completion(usage=total, mode=mode, truncated=truncated,
                                          error=f"no JSON in response: {error}")
                return Completion(text=text, usage=total, truncated=truncated, mode=mode)

            reasoning = getattr(choice.message, "reasoning_content", None) or ""
            if truncated:
                detail = f" after {len(reasoning)} chars of reasoning" if reasoning else ""
                return Completion(
                    usage=total, mode=mode, truncated=True,
                    error=f"empty response: token budget ran out{detail}. Raise DI_MAX_TOKENS.")

            # Content is empty but generation stopped normally. That is what a model
            # whose engine cannot really honour json_schema looks like: the request is
            # accepted and decoding yields nothing. Fall through to prompt mode rather
            # than reporting a failure that is actually a capability gap.
            if mode == "schema" and self.json_mode == "auto" and attempt == 0:
                self._downgrade()
                continue
            detail = f" ({len(reasoning)} chars of reasoning)" if reasoning else ""
            return Completion(usage=total, mode=mode,
                              error=f"empty response: no content returned{detail}")
        return Completion(usage=total, error="empty response")


# ------------------------------------------------------------------ selection
BACKENDS = {"anthropic": AnthropicBackend, "openai": OpenAIBackend}
ALIASES = {"openai-compatible": "openai", "lmstudio": "openai", "local": "openai",
           "claude": "anthropic"}


def build(config=None, plugin: str = "", overrides=None):
    """Construct the configured extractor backend.

    The plugin is chosen once -- in the manifest, or by an explicit override -- and its
    settings come from its own block. Nothing else has to be hunted down.
    """
    from core import config as config_mod
    from core.plugins import SettingsError, cross_plugin_hint

    config = config or config_mod.load()
    chosen = (config.chosen("extractor", plugin) or "openai").strip().lower()
    chosen = ALIASES.get(chosen, chosen)
    if chosen not in BACKENDS:
        raise SystemExit(
            f"unknown extractor {chosen!r}; available: {', '.join(sorted(BACKENDS))}")

    backend_cls = BACKENDS[chosen]
    try:
        settings = config.settings("extractor", chosen, backend_cls.SETTINGS, overrides)
    except SettingsError as error:
        message = str(error)
        # If the key is real but belongs to the other backend, say which one.
        for line in message.splitlines():
            if " has no setting " in line:
                key = line.split(" has no setting ")[1].strip().rstrip(".").strip("'\"")
                others = {n: c.SETTINGS for n, c in BACKENDS.items() if n != chosen}
                hint = cross_plugin_hint(key, others)
                if hint:
                    message += f"\n  {hint}"
                break
        raise SystemExit(f"configuration error: {message}")

    if chosen == "openai" and not settings.get("model"):
        probe = OpenAIBackend(model="unset", **{k: v for k, v in settings.items()
                                                if k != "model"})
        names = probe.available_models()
        listing = "\n".join(f"  {n}" for n in names) if names else \
            "  (could not reach the endpoint to list them)"
        raise SystemExit(
            f"no model set for the openai extractor. Available at "
            f"{settings['base_url']}:\n{listing}")

    backend = backend_cls(**settings)
    backend.settings = settings
    backend.provenance = config.provenance("extractor", chosen, backend_cls.SETTINGS,
                                           settings)
    return backend
