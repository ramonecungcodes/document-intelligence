"""The pipeline manifest: which plugin fills each slot, and how each is configured.

One file says what the pipeline is made of. Swapping the extractor and re-running the
benchmark is a one-line diff here rather than three environment variables and hoping
you picked the right ones -- which is the whole argument for having plugin seams.

TOML, read with the standard library. The manifest is committable: it holds no
secrets, only `${VAR}` references that resolve from the environment at load.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from typing import Optional

from core.plugins import SettingsError, bind, redact

DEFAULT_PATHS = ("di.toml", "/app/di.toml")

# A slot is a stage of the pipeline. Only `extractor` is filled today; the rest are
# named here so the manifest has somewhere obvious to grow into.
SLOTS = ("source", "normalizer", "splitter", "classifier", "extractor", "validator", "sink")


@dataclass
class Config:
    path: Optional[str] = None
    pipeline: dict = field(default_factory=dict)
    blocks: dict = field(default_factory=dict)      # slot -> plugin -> settings block

    def chosen(self, slot: str, override: str = "") -> Optional[str]:
        """Which plugin fills a slot: an explicit override, then the manifest."""
        if override:
            return override
        env = os.environ.get(f"DI_{slot.upper()}")
        return env or self.pipeline.get(slot)

    def block(self, slot: str, plugin: str) -> dict:
        return dict(self.blocks.get(slot, {}).get(plugin, {}))

    def settings(self, slot: str, plugin: str, specs, overrides=None) -> dict:
        return bind(plugin, specs, self.block(slot, plugin), os.environ, overrides)

    def provenance(self, slot: str, plugin: str, specs, settings: dict) -> dict:
        """What produced a score, safe to write into a report."""
        return {
            "slot": slot,
            "plugin": plugin,
            "manifest": self.path,
            "settings": redact(settings, specs),
        }


def find_manifest(explicit: str = "") -> Optional[str]:
    for candidate in ([explicit] if explicit else []) + [os.environ.get("DI_CONFIG", "")] + \
            list(DEFAULT_PATHS):
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def load(path: str = "") -> Config:
    """Read the manifest. Absent is fine -- everything falls back to defaults."""
    found = find_manifest(path)
    if not found:
        return Config()
    try:
        with open(found, "rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise SettingsError(f"{found} is not valid TOML: {error}")

    pipeline = data.get("pipeline", {})
    unknown = [slot for slot in pipeline if slot not in SLOTS]
    if unknown:
        raise SettingsError(
            f"{found}: [pipeline] has no slot {unknown[0]!r}; known slots are "
            f"{', '.join(SLOTS)}")

    blocks = {}
    for slot in SLOTS:
        # Plural section name reads better in the file: [extractors.openai]
        section = data.get(slot + "s", {})
        if isinstance(section, dict):
            blocks[slot] = {name: value for name, value in section.items()
                            if isinstance(value, dict)}
    return Config(path=found, pipeline=pipeline, blocks=blocks)
