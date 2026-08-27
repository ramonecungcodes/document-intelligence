"""The two do-nothing splitters, which is what every real one is measured against.

Phase 0 established that a number needs an anchor, and this stage needs two of them
because it can fail in opposite directions.

`single` never splits. It is what the pipeline did before this stage existed, and on a
corpus where most files hold one document it scores respectably while being useless.

`every_page` splits everywhere. It finds every boundary there is -- perfect recall, by
construction -- and is wrong about every interior page of a multi-page document.

Between them they bracket the problem. A splitter that cannot beat both is not earning
its place, and quoting a splitter's F1 without them is quoting a number with no scale.
"""
from __future__ import annotations

import time

from core.plugins import Setting
from split.base import Split, register


def page_count(path: str) -> int:
    from normalize.native import open_pdf
    document = open_pdf(path)
    try:
        return document.page_count
    finally:
        document.close()


@register("single")
class Single:
    """Treat the whole file as one document."""

    SETTINGS = ()

    def describe(self) -> str:
        return "single - the file is one document"

    def split(self, path: str, **_) -> Split:
        started = time.time()
        return Split(boundaries=[], pages=page_count(path), engine="single",
                     seconds=time.time() - started)


@register("every_page")
class EveryPage:
    """Treat every page as its own document."""

    SETTINGS = ()

    def describe(self) -> str:
        return "every_page - each page is a document"

    def split(self, path: str, **_) -> Split:
        started = time.time()
        pages = page_count(path)
        return Split(boundaries=list(range(1, pages)), pages=pages,
                     engine="every_page", seconds=time.time() - started)
