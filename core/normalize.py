"""Field normalisation and comparison primitives.

These live in `core` rather than `eval` on purpose. Deciding whether `03/29/2026`
is the same date as `2026-03-29`, or `$4,102.50` the same amount as `4102.5`, is a
question the validator plugins will ask in production for exactly the same reason
the scorer asks it now. One implementation, so the two can never drift apart and
report a field correct in evaluation but invalid in the pipeline.

Nothing here reaches for a third-party library: `difflib` handles the fuzzy case and
`datetime.strptime` handles the date formats the corpus actually renders.
"""
from __future__ import annotations

import datetime
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

# Formats the generator renders, plus the ISO form the labels store.
DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%Y/%m/%d",
)

# Stripped before comparing company names: a vendor is the same vendor whether or
# not the extractor kept the legal suffix.
LEGAL_SUFFIXES = (
    "incorporated", "inc", "llc", "l l c", "llp", "l l p", "ltd", "limited",
    "corporation", "corp", "co", "company", "plc", "pvt", "private", "gmbh", "sa", "nv",
)

MONEY_CHARS = re.compile(r"[^\d.\-()]")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
WS = re.compile(r"\s+")

DEFAULT_MONEY_TOLERANCE = 0.01
DEFAULT_NAME_THRESHOLD = 0.90
DEFAULT_TEXT_THRESHOLD = 0.85


@dataclass(frozen=True)
class Comparison:
    """The outcome of comparing one predicted value against ground truth.

    `exact` is the strict reading and `match` the one that counts; reporting both is
    what lets the scorer say "+3 recovered by date normalisation" rather than hiding
    the difference between a real improvement and a lenient comparison.
    """

    exact: bool
    match: bool
    note: str = ""

    @property
    def normalised_only(self) -> bool:
        return self.match and not self.exact


def is_blank(value) -> bool:
    """Absent, empty, or whitespace. The corpus writes "" for fields a defect removed."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _clean(value) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return WS.sub(" ", text).strip()


# ------------------------------------------------------------------ parsers
def parse_date(value):
    """Return a `date`, or None if nothing recognisable is in there."""
    if is_blank(value):
        return None
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.datetime):
        return value.date()
    text = _clean(value)
    for fmt in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_money(value):
    """Return a float, or None. Handles `$1,234.56`, `INR 1,234.00`, `(45.00)`."""
    if is_blank(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = _clean(value)
    negative = text.startswith("(") and text.endswith(")")
    digits = MONEY_CHARS.sub("", text).replace("(", "").replace(")", "")
    if digits in ("", "-", "."):
        return None
    try:
        amount = float(digits)
    except ValueError:
        return None
    return -amount if negative and amount > 0 else amount


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if is_blank(value):
        return None
    text = _clean(value).lower()
    if text in ("true", "yes", "y", "1", "checked", "x", "☑"):
        return True
    if text in ("false", "no", "n", "0", "unchecked", "☐"):
        return False
    return None


def digits_only(value) -> str:
    return re.sub(r"\D", "", _clean(value))


def normalise_identifier(value) -> str:
    """Case and separators vary with how the document was read; the identity does not."""
    return NON_ALNUM.sub("", _clean(value).lower())


def normalise_text(value) -> str:
    return NON_ALNUM.sub(" ", _clean(value).lower()).strip()


def normalise_name(value) -> str:
    """Drop punctuation and legal suffixes so `Acme, Inc.` matches `Acme Inc`.

    Trailing phrases are checked as well as single words: punctuation stripping turns
    `L.L.C.` into `l l c`, which no single-word check would recognise.
    """
    words = normalise_text(value).split()
    changed = True
    while changed and words:
        changed = False
        for span in (3, 2, 1):
            if len(words) >= span and " ".join(words[-span:]) in LEGAL_SUFFIXES:
                del words[-span:]
                changed = True
                break
    return " ".join(words)


def similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


# ------------------------------------------------------------------ comparison
def _both_blank(predicted, truth):
    """A field a defect emptied is correctly extracted as empty."""
    return Comparison(True, True, "both blank") if is_blank(predicted) and is_blank(truth) else None


def _one_blank(predicted, truth):
    if is_blank(truth):
        return Comparison(False, False, "predicted a value where truth is blank")
    if is_blank(predicted):
        return Comparison(False, False, "missing")
    return None


def compare(kind: str, predicted, truth, *, tolerance=None, threshold=None) -> Comparison:
    """Compare one value under the rules for its declared field kind."""
    blank = _both_blank(predicted, truth)
    if blank:
        return blank
    blank = _one_blank(predicted, truth)
    if blank:
        return blank

    exact = _clean(predicted) == _clean(truth)

    if kind == "date":
        p, t = parse_date(predicted), parse_date(truth)
        if p is None or t is None:
            return Comparison(exact, exact, "unparseable date" if not exact else "")
        return Comparison(exact, p == t, "" if exact else "date format")

    if kind in ("money", "number"):
        tol = DEFAULT_MONEY_TOLERANCE if tolerance is None else tolerance
        p, t = parse_money(predicted), parse_money(truth)
        if p is None or t is None:
            return Comparison(exact, exact, "unparseable number" if not exact else "")
        return Comparison(exact, abs(p - t) <= tol, "" if exact else f"within {tol}")

    if kind == "bool":
        p, t = parse_bool(predicted), parse_bool(truth)
        if p is None or t is None:
            return Comparison(exact, exact, "unparseable boolean" if not exact else "")
        return Comparison(exact, p == t, "" if exact else "boolean form")

    if kind in ("identifier", "enum"):
        match = normalise_identifier(predicted) == normalise_identifier(truth)
        return Comparison(exact, match, "" if exact else "case/separators")

    if kind in ("ssn", "ein", "phone", "account"):
        match = digits_only(predicted) == digits_only(truth)
        return Comparison(exact, match, "" if exact else "formatting")

    if kind == "email":
        match = _clean(predicted).lower() == _clean(truth).lower()
        return Comparison(exact, match, "" if exact else "case")

    if kind == "name":
        thr = DEFAULT_NAME_THRESHOLD if threshold is None else threshold
        ratio = similarity(normalise_name(predicted), normalise_name(truth))
        return Comparison(exact, ratio >= thr, "" if exact else f"fuzzy {ratio:.2f}")

    # default: free text
    thr = DEFAULT_TEXT_THRESHOLD if threshold is None else threshold
    ratio = similarity(normalise_text(predicted), normalise_text(truth))
    return Comparison(exact, ratio >= thr, "" if exact else f"fuzzy {ratio:.2f}")
