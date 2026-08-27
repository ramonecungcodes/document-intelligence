"""Read text a previous normalizer run already produced.

This is what the extractor uses once OCR has been done. It is a normalizer like any
other, which is the whole reason the seam was worth having: the extractor asks for text
and gets text, and never learns whether an OCR engine ran a second ago or a week ago.

`engine` selects which cached tree to read, so comparing two engines through the
extractor is a one-line manifest change over pre-computed text -- no GPU, no OCR
dependency, and no risk that a model comparison is quietly measuring OCR variance
instead.

A missing entry is an error by default rather than a silent fall-through to reading the
PDF. A cache that quietly half-populates would produce a score blending OCR text with
native text and no way to tell which documents were which, and that is precisely the
kind of unattributable number this project exists not to produce.
"""
from __future__ import annotations

import os

from core.plugins import Setting
from normalize.base import Extracted, register
from normalize import store

DEFAULT_ROOT = os.environ.get("DI_NORMALIZED_DIR", "/data/normalized")


@register("cached")
class Cached:
    """Serve text from a normalizer run that already happened."""

    SETTINGS = (
        Setting("root", str, default=DEFAULT_ROOT, help="where normalized text was written"),
        Setting("engine", str, default="tesseract", help="which engine's output to read"),
        Setting("corpus", str, default="",
                help="corpus root, to turn an absolute document path back into its key"),
        Setting("on_missing", str, default="error",
                help="error | empty — what to do when a document was never normalized"),
    )

    def __init__(self, root: str = DEFAULT_ROOT, engine: str = "tesseract",
                 corpus: str = "", on_missing: str = "error", **_):
        self.root = root
        self.name = engine
        self.corpus = corpus or os.environ.get("DI_DATASET_ROOT", "/data")
        self.on_missing = on_missing

    def describe(self) -> str:
        return f"cached · {self.name} · {self.root}"

    def key_for(self, path: str) -> str:
        """The corpus-relative path the cache is keyed on."""
        absolute = os.path.abspath(path)
        root = os.path.abspath(self.corpus)
        if absolute.startswith(root + os.sep):
            return os.path.relpath(absolute, root).replace("\\", "/")
        return os.path.basename(path)

    def read(self, path: str) -> Extracted:
        key = self.key_for(path)
        if not store.exists(self.root, self.name, key):
            if self.on_missing == "empty":
                return Extracted(text="", pages=0, layer="none",
                                 engine=f"cached:{self.name}:missing")
            raise SystemExit(
                f"no normalized text for {key!r} under engine {self.name!r} in "
                f"{self.root}.\n"
                f"  Run the normalizer first:  docker compose run --rm normalizer "
                f"run --engine {self.name}")
        result = store.read(self.root, self.name, key)
        result.engine = f"cached:{result.engine}"
        return result
