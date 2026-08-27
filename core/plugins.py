"""Plugin settings: each plugin declares what it needs, the core binds and checks it.

The problem this solves is a flat configuration namespace. With ten variables shared
across two plugins, most of them inert depending on a choice made in an eleventh,
setting `effort` on a backend that has no notion of effort does nothing at all -- no
error, no warning, just a value that was silently never read. That failure mode gets
worse with every plugin added.

So a plugin owns its settings. Choose the plugin and everything it needs is in one
block, unknown keys are errors rather than silence, and a setting that belongs to a
sibling plugin says so by name.

Secrets are the one deliberate exception. A setting marked `secret=True` is written in
the manifest as `${SOME_ENV_VAR}` and resolved at load time, so the manifest stays
committable and the token never does. Secrets are also redacted from run provenance,
which is what makes it safe to record the full resolved configuration alongside a
score.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Optional

ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
REDACTED = "<set>"


class SettingsError(Exception):
    """Raised for a configuration mistake, with a message aimed at the person."""


@dataclass(frozen=True)
class Setting:
    name: str
    kind: type = str
    default: Any = None
    required: bool = False
    secret: bool = False
    help: str = ""


def _coerce(setting: Setting, value: Any, plugin: str) -> Any:
    if value is None:
        return None
    if setting.kind is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        raise SettingsError(f"{plugin}.{setting.name}: expected true or false, got {value!r}")
    if setting.kind in (int, float):
        try:
            return setting.kind(value)
        except (TypeError, ValueError):
            raise SettingsError(
                f"{plugin}.{setting.name}: expected a {setting.kind.__name__}, got {value!r}")
    return str(value)


def _resolve_env(value: Any, env: dict) -> Any:
    """Turn `${VAR}` into the environment's value, or None when it is unset."""
    if isinstance(value, str):
        match = ENV_REF.match(value.strip())
        if match:
            return env.get(match.group(1)) or None
    return value


def env_key(plugin: str, setting: str) -> str:
    """The namespaced override for one setting: DI_OPENAI_MODEL, DI_ANTHROPIC_EFFORT."""
    clean = re.sub(r"[^A-Za-z0-9]+", "_", plugin).strip("_").upper()
    return f"DI_{clean}_{setting.upper()}"


def bind(plugin: str, specs, block: Optional[dict] = None,
         env: Optional[dict] = None, overrides: Optional[dict] = None) -> dict:
    """Resolve one plugin's settings.

    Precedence, lowest to highest: declared default, manifest block, namespaced
    environment variable, explicit override (a command-line flag).
    """
    env = os.environ if env is None else env
    block = dict(block or {})
    by_name = {spec.name: spec for spec in specs}

    unknown = [key for key in block if key not in by_name]
    if unknown:
        raise SettingsError(_unknown_message(plugin, unknown[0], by_name))

    resolved = {}
    for spec in specs:
        value = spec.default
        if spec.name in block:
            value = _resolve_env(block[spec.name], env)
        override_env = env.get(env_key(plugin, spec.name))
        if override_env not in (None, ""):
            value = override_env
        if overrides and overrides.get(spec.name) not in (None, ""):
            value = overrides[spec.name]

        value = _coerce(spec, value, plugin)
        if spec.required and (value is None or value == ""):
            raise SettingsError(_missing_message(plugin, spec))
        resolved[spec.name] = value
    return resolved


def _unknown_message(plugin: str, key: str, by_name: dict) -> str:
    near = get_close_matches(key, list(by_name), n=1, cutoff=0.7)
    hint = f" Did you mean {near[0]!r}?" if near else ""
    known = ", ".join(sorted(by_name)) or "(none)"
    return (f"{plugin!r} has no setting {key!r}.{hint}\n"
            f"  it accepts: {known}")


def _missing_message(plugin: str, spec: Setting) -> str:
    where = env_key(plugin, spec.name)
    detail = f"  {spec.help}\n" if spec.help else ""
    source = (f"  set it in the manifest as {spec.name} = \"${{{where}}}\" "
              f"and export {where}"
              if spec.secret else
              f"  set it in the manifest under [{plugin}] or export {where}")
    return f"{plugin!r} requires {spec.name!r}.\n{detail}{source}"


def cross_plugin_hint(key: str, plugins: dict) -> Optional[str]:
    """Whether an unknown key is a real setting on a sibling plugin.

    This is the exact mistake the flat namespace invited: `effort` is meaningful, just
    not here. Saying so is more useful than listing valid keys.
    """
    for name, specs in plugins.items():
        if any(spec.name == key for spec in specs):
            return f"{key!r} is a setting of the {name!r} plugin, not this one"
    return None


def redact(settings: dict, specs) -> dict:
    """The resolved settings with secrets masked, safe to record beside a score."""
    secret = {spec.name for spec in specs if spec.secret}
    return {
        name: (REDACTED if (name in secret and value) else value)
        for name, value in settings.items()
    }


def describe(plugin: str, specs) -> str:
    """Human-readable settings reference for one plugin."""
    lines = [f"[{plugin}]"]
    for spec in sorted(specs, key=lambda s: (not s.required, s.name)):
        bits = [spec.kind.__name__]
        if spec.required:
            bits.append("required")
        if spec.secret:
            bits.append("secret")
        if spec.default not in (None, ""):
            bits.append(f"default {spec.default!r}")
        lines.append(f"  {spec.name:<12} {', '.join(bits)}")
        if spec.help:
            lines.append(f"  {'':<12} {spec.help}")
    return "\n".join(lines)
