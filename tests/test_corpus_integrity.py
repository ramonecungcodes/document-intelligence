"""Ground truth must describe the document.

A label that claims a value the page never shows is an unanswerable question. It caps
that field's accuracy at whatever fraction of layouts happen to render it, and the
shortfall looks exactly like model weakness -- multi-bill `service_code` sat at 0.000
across 84 rows because two of its three layouts never printed it.

These tests read the rendered PDFs, so they need a generated corpus and skip without
one. They are the check that should have run the day the generator was written.
"""
import json
import os

import pytest

from extract.text import read_pdf

CORPUS = os.environ.get("DI_DATASET_ROOT", "/data")

# Values that legitimately do not appear as printed text.
NOT_RENDERED = {
    "file", "doc_type", "layout", "irregularities", "source_file", "degradation",
    "section_index",     # positional, not printed
    "currency",          # shown as a symbol, not the code
    "reference_label",   # printed, but as a word that may repeat; checked with its number
}

# Types whose labels are prose-derived rather than transcribed verbatim.
SKIP_TYPES = {"resumes"}


def corpus_records(stem):
    path = os.path.join(CORPUS, "labels", f"{stem}.json")
    if not os.path.exists(path):
        pytest.skip(f"no generated corpus at {path}")
    return json.load(open(path, encoding="utf-8"))


def rendered_text(record):
    pdf = os.path.join(CORPUS, record["file"])
    if not os.path.exists(pdf):
        pytest.skip(f"{record['file']} not rendered")
    return read_pdf(pdf).text


def appears(value, text) -> bool:
    """Is this labelled value findable in the rendered text?

    Dates need every rendered form tried: labels store ISO but documents print
    `Apr 21, 2026` or `04/21/2026`, deliberately, so that extraction has to normalise.
    Comparing raw strings would flag the corpus's whole point as a defect.
    """
    from core.normalize import parse_date
    if isinstance(value, float):
        return f"{value:,.2f}" in text or f"{value:.2f}" in text
    text_value = str(value)
    parsed = parse_date(text_value)
    if parsed is not None and len(text_value) == 10 and text_value[4] == "-":
        return any(parsed.strftime(fmt) in text for fmt in
                   ("%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y", "%-m/%-d/%Y"))
    return text_value in text


def scalar_values(record):
    """Every labelled scalar that should be findable in the document text."""
    for key, value in record.items():
        if key in NOT_RENDERED or isinstance(value, (list, dict)):
            continue
        if value in (None, "", []):
            continue
        yield key, value


class TestMultiBillSectionsAreRendered:
    """The defect that prompted this file: identifiers labelled but not printed."""

    @pytest.mark.parametrize("layout", [0, 1, 2])
    @pytest.mark.parametrize("field", ["service_code", "account_number", "reference_number"])
    def test_section_identifier_appears_on_every_layout(self, layout, field):
        records = [r for r in corpus_records("multi_bill_invoices") if r["layout"] == layout]
        if not records:
            pytest.skip(f"no layout {layout} documents")
        missing = []
        for record in records[:6]:
            text = rendered_text(record)
            for section in record["sections"]:
                value = section.get(field)
                if value and str(value) not in text:
                    missing.append(f"{os.path.basename(record['file'])}:{value}")
        assert not missing, (
            f"layout {layout} labels {field} but does not print it: {missing[:3]}"
        )


class TestHeaderFieldsAreRendered:
    @pytest.mark.parametrize("stem", ["invoices", "purchase_orders", "multi_bill_invoices"])
    def test_labelled_scalars_appear_in_the_document(self, stem):
        records = corpus_records(stem)
        missing = []
        for record in records[:8]:
            text = rendered_text(record)
            for key, value in scalar_values(record):
                if not appears(value, text):
                    missing.append(f"{os.path.basename(record['file'])}.{key}={value!r}")
        assert not missing, f"{stem} labels values the page does not show: {missing[:5]}"


class TestCleanCorpusIsClean:
    @pytest.mark.parametrize("stem", ["invoices", "purchase_orders", "multi_bill_invoices"])
    def test_no_defects_tagged_in_the_clean_set(self, stem):
        assert not [r for r in corpus_records(stem) if r.get("irregularities")]

    def test_multi_bill_sections_roll_up(self):
        for record in corpus_records("multi_bill_invoices"):
            total = round(sum(s["total"] for s in record["sections"]), 2)
            assert abs(total - record["total"]) < 0.02, record["file"]
