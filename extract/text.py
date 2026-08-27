"""Get text out of a PDF.

This is now a thin front for the `normalizer` stage. Phase 1 read the embedded text
layer and nothing else; that reader still exists and is still the default, but it lives
in `normalize/` alongside the OCR plugins that Phase 2 measures it against.

Documents with no text layer -- the degraded scans -- still come back empty under the
default normalizer, and that is still the honest result rather than a bug to paper
over: it is the measured size of the gap OCR has to close.

`read_pdf` is kept because a great deal of code and several tests call it, and because
"read this PDF the default way" is a genuinely useful thing to say. Anything that cares
which engine ran should take a normalizer instead.
"""
from __future__ import annotations

from normalize.base import Extracted  # noqa: F401  (re-exported; callers import it here)
from normalize.native import NativeText

_DEFAULT = NativeText()


def read_pdf(path: str) -> Extracted:
    """Read a PDF with the default normalizer: the embedded text layer."""
    return _DEFAULT.read(path)
