"""Schema generation and PDF text reading — everything that needs no API call."""
import json
import os

import pytest

from core.doctypes import INVOICE, MULTI_BILL_INVOICE, REGISTRY
from extract.schema import instructions, json_schema
from extract.text import read_pdf

SAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "document-generator", "samples",
)


class TestSchema:
    @pytest.mark.parametrize("name", sorted(REGISTRY))
    def test_every_type_produces_a_valid_structured_output_schema(self, name):
        """Structured outputs rejects a schema that omits either of these."""
        schema = json_schema(REGISTRY[name])
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        json.dumps(schema)                       # must be serialisable

    def test_every_field_is_nullable(self):
        """A W-9 has an SSN or an EIN, never both; the model needs a way to say so
        without inventing a value or dropping the key."""
        for prop in json_schema(REGISTRY["form"])["properties"].values():
            if prop.get("type") and isinstance(prop["type"], list):
                assert "null" in prop["type"]

    def test_money_is_a_number_and_dates_stay_strings(self):
        props = json_schema(INVOICE)["properties"]
        assert props["total"]["type"] == ["number", "null"]
        # Reformatting a date is a second chance to get it wrong.
        assert props["invoice_date"]["type"] == ["string", "null"]

    def test_repeating_groups_become_arrays_of_objects(self):
        props = json_schema(INVOICE)["properties"]
        assert props["line_items"]["type"] == "array"
        item = props["line_items"]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == {"description", "quantity", "unit_price", "amount"}

    def test_nested_groups_survive(self):
        """Sections contain line items; both levels have to be in the schema."""
        sections = json_schema(MULTI_BILL_INVOICE)["properties"]["sections"]["items"]
        assert "account_number" in sections["properties"]
        assert sections["properties"]["line_items"]["type"] == "array"

    def test_schema_matches_the_scorer_field_set(self):
        """Extractor and scorer read the same declaration, so they cannot drift."""
        for name, doctype in REGISTRY.items():
            declared = {f.name for f in doctype.fields} | {g.name for g in doctype.groups}
            assert set(json_schema(doctype)["properties"]) == declared, name

    def test_multi_bill_prompt_calls_out_separate_payment(self):
        text = instructions(MULTI_BILL_INVOICE)
        assert "paid separately" in text
        assert "sections" in text

    def test_prompt_tells_the_model_not_to_correct_the_document(self):
        """Transcribing a wrong total is correct behaviour; the validators find it."""
        assert "even when it looks wrong" in instructions(INVOICE)


class TestText:
    def test_reads_a_native_text_layer(self):
        result = read_pdf(os.path.join(SAMPLES, "multi-bill-invoice.pdf"))
        assert result.layer == "native"
        assert not result.empty
        assert "INVOICE" in result.text.upper()
        assert result.pages >= 1

    def test_multi_page_documents_are_marked_up_by_page(self):
        result = read_pdf(os.path.join(SAMPLES, "multi-bill-invoice.pdf"))
        if result.pages > 1:
            assert "--- page 2 ---" in result.text

    def test_a_scanned_document_reports_no_text_layer(self):
        """Not a failure to fix -- the measured size of the gap OCR has to close."""
        result = read_pdf(os.path.join(SAMPLES, "multi-bill-invoice.scanned.pdf"))
        assert result.layer == "none"
        assert result.empty

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_pdf(os.path.join(SAMPLES, "does-not-exist.pdf"))
