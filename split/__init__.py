"""Splitters: work out how many documents a file actually holds.

Every stage after this one is told "this is one document". When that is wrong the
classifier types a chimera and the extractor reads fields off pages belonging to two
different documents, so an error here does not degrade the output, it invents an
output for a document that never existed.

    single      the file is one document -- what the pipeline did before this stage
    every_page  each page is a document -- perfect recall, by construction
    by_type     classify each page, cut where the answer changes

The two baselines bracket the problem from opposite sides and both are always
reported, because a boundary F1 quoted on its own has no scale.
"""
from split.base import SPLITTERS, Split, build  # noqa: F401
from split import baselines, by_type  # noqa: F401  (registration side effect)

__all__ = ["Split", "SPLITTERS", "build"]
