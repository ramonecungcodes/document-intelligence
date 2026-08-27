"""Normalizers built out of other normalizers.

A cascade is not a new kind of pipeline stage, and neither is an ensemble. Both are
normalizers that happen to delegate, so both fill the slot the same way everything else
does. Ordering lives in the composite's own settings rather than being smeared across
the pipeline definition, and `build()` needs no special case -- which is the point of
having one contract per slot instead of a core with exceptions bolted on.

    cascade     cheap engine first; escalate only where its confidence is poor
    consensus   run every engine, keep the best, and report whether they agreed

Both record what actually ran. A score attributed to "cascade" is unattributable: if it
escalated on 12% of documents you cannot tell which engine earned the number. So
`engine` always names the engine whose text was kept, and the decision that led there.
"""
from __future__ import annotations

import time

from core.plugins import Setting
from normalize.base import Extracted, register


def _members(names, overrides=None):
    """Build each named normalizer from its own settings block."""
    from normalize.base import build
    if isinstance(names, str):
        names = [part.strip() for part in names.split(",") if part.strip()]
    if len(names) < 2:
        raise SystemExit("a composite normalizer needs at least two engines")
    return [(name, build(plugin=name, overrides=overrides)) for name in names]


@register("cascade")
class Cascade:
    """Try engines in order; stop at the first whose confidence clears the bar.

    The economics are the argument. OCR cost is dominated by the expensive engine, and
    most documents do not need it -- a clean scan is read correctly by anything. Paying
    for the neural model only where the classical one reports it struggled means the
    average document costs almost nothing and the hard ones still get read.

    Escalation triggers on the engine's own word confidence, deliberately, and never on
    anything derived from extraction. Coupling the stages would make the OCR number
    depend on the model and destroy the isolation this stage exists to provide.
    """

    SETTINGS = (
        Setting("engines", str, default="native,tesseract,doctr",
                help="comma-separated, cheapest first"),
        Setting("escalate_below", float, default=0.80,
                help="mean word confidence under which the next engine is tried (0-1)"),
        Setting("min_chars", int, default=40,
                help="text shorter than this counts as a failure regardless of confidence"),
    )

    def __init__(self, engines: str = "native,tesseract,doctr",
                 escalate_below: float = 0.80, min_chars: int = 40, **_):
        self.names = engines
        self.escalate_below = escalate_below
        self.min_chars = min_chars
        self._members = None

    def describe(self) -> str:
        return f"cascade · {self.names} · escalate below {self.escalate_below}"

    def members(self):
        if self._members is None:
            self._members = _members(self.names)
        return self._members

    def _good_enough(self, result: Extracted) -> bool:
        if len(result.text.strip()) < self.min_chars:
            return False
        # A native text layer is exact by construction; no engine will improve on it,
        # and it reports no confidence to compare against.
        if result.layer == "native":
            return True
        return result.confidence is not None and result.confidence >= self.escalate_below

    def read(self, path: str) -> Extracted:
        started = time.time()
        attempts, results, accepted = [], [], None
        for name, engine in self.members():
            result = engine.read(path)
            attempts.append(f"{name}={_summarise(result)}")
            results.append(result)
            if self._good_enough(result):
                accepted = result
                break

        # Only a result that cleared the bar wins outright. If none did, fall back to
        # the least-bad -- but rank it with the same length floor, because an engine
        # reporting 0.99 confidence in three characters has not read the document, and
        # ranking on confidence alone would let that fragment beat a full page.
        best = accepted if accepted is not None else max(
            results, key=lambda r: _rank(r, self.min_chars))

        best.engine = f"cascade:{best.engine}"
        best.seconds = time.time() - started
        best.tried = attempts
        return best


@register("consensus")
class Consensus:
    """Run every engine, keep the most confident, and report whether they agreed.

    Agreement between two independently-built OCR engines is a real confidence signal.
    It is not a model's opinion of its own output -- which is the thing calibration must
    not be built on -- but two systems that share no weights arriving at the same
    string. Where they agree, trust is earned cheaply. Where they diverge, the hard
    region on the page has located itself.

    The agreement figure here is deliberately coarse: a similarity ratio over the
    normalised text. Two engines will disagree about whitespace, line breaks and reading
    order long before they disagree about a digit, so anything stricter would report
    disagreement that does not matter. Field-level comparison belongs downstream, where
    it can be done against the extracted values rather than the raw page.
    """

    SETTINGS = (
        Setting("engines", str, default="tesseract,doctr",
                help="comma-separated; all are run on every document"),
    )

    def __init__(self, engines: str = "tesseract,doctr", **_):
        self.names = engines
        self._members = None

    def describe(self) -> str:
        return f"consensus · {self.names}"

    def members(self):
        if self._members is None:
            self._members = _members(self.names)
        return self._members

    def read(self, path: str) -> Extracted:
        from core.normalize import similarity, normalise_text

        started = time.time()
        results = [(name, engine.read(path)) for name, engine in self.members()]
        best = max((r for _, r in results), key=_rank)   # no length floor to apply here

        agreement = None
        texts = [normalise_text(r.text) for _, r in results if r.text.strip()]
        if len(texts) >= 2:
            pairs = [(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]]
            agreement = sum(similarity(a, b) for a, b in pairs) / len(pairs)

        best.engine = f"consensus:{best.engine}"
        best.seconds = time.time() - started
        best.agreement = agreement                      # noqa: provenance
        best.tried = [f"{n}={_summarise(r)}" for n, r in results]
        return best


def _rank(result: Extracted, min_chars: int = 0):
    """Which of two results to keep.

    Tiered rather than weighted, because the tiers are not commensurable. Text at all
    beats no text; a plausible amount of text beats a fragment however confident the
    engine was about it; only then does confidence decide.
    """
    body = result.text.strip()
    return (bool(body),
            len(body) >= min_chars,
            result.confidence if result.confidence is not None else 0.0,
            len(body))


def _summarise(result: Extracted) -> str:
    confidence = "-" if result.confidence is None else f"{result.confidence:.2f}"
    return f"{len(result.text)}ch/conf{confidence}/{result.seconds:.1f}s"
