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
            declared = {f.name for f in doctype.fields_for()} | {g.name for g in doctype.groups}
            assert set(json_schema(doctype)["properties"]) == declared, name
            for variant in doctype.variants:
                expected = {f.name for f in doctype.fields_for(variant)} |                            {g.name for g in doctype.groups}
                assert set(json_schema(doctype, variant)["properties"]) == expected,                     f"{name}/{variant}"

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


class TestSchemaRejectionFallback:
    """A 400 on the json_schema request is a downgrade signal, not a dead document.

    Endpoints disagree about what schema they will accept. LM Studio refuses a
    nullable field's `["string", "null"]` outright for some models, which made an
    entire model unusable: `auto` mode listed prompt as its fallback but the exception
    path returned before ever trying it.
    """

    def error(self, status, message="response_format schema rejected"):
        class Rejected(Exception):
            status_code = status
        return Rejected(message)

    def test_the_real_rejection_this_was_built_for(self):
        """LM Studio's refusal of a nullable field's ["string", "null"]."""
        from extract.backends import _is_schema_rejection
        assert _is_schema_rejection(self.error(
            400, "Error in iterating prediction stream: ValueError: 'type' must be a string"))

    def test_a_response_format_complaint_is_a_rejection(self):
        from extract.backends import _is_schema_rejection
        assert _is_schema_rejection(
            self.error(400, "response_format json_schema is not supported"))

    def test_a_context_length_400_is_not(self):
        """The request was too big, not badly shaped; prompt mode would only be bigger."""
        from extract.backends import _is_schema_rejection
        assert not _is_schema_rejection(
            self.error(400, "This model's maximum context length is 8192 tokens"))

    def test_a_rate_limit_400_is_not(self):
        from extract.backends import _is_schema_rejection
        assert not _is_schema_rejection(self.error(400, "Rate limit exceeded"))

    def test_a_model_that_will_not_load_is_not(self):
        """Observed in a real run: this downgraded, wasted a retry, and pinned the
        rest of the run to prompt mode for a reason unrelated to the schema."""
        from extract.backends import _is_schema_rejection
        assert not _is_schema_rejection(self.error(
            400, 'Failed to load model "google/gemma-4-e4b". Error: Model loading was '
                 "stopped due to insufficient system resources."))

    def test_an_unrecognised_400_does_not_downgrade(self):
        """Unknown means fail with the real error, not guess and retry."""
        from extract.backends import _is_schema_rejection
        assert not _is_schema_rejection(self.error(400, "something else went wrong"))

    def test_other_statuses_are_not(self):
        from extract.backends import _is_schema_rejection
        for status in (401, 404, 429, 500, 503, None):
            assert not _is_schema_rejection(self.error(status)), status

    def test_a_timeout_is_not(self):
        from extract.backends import _is_schema_rejection
        assert not _is_schema_rejection(TimeoutError("timed out"))

    def build(self, effects, json_mode="auto"):
        from extract.backends import OpenAIBackend
        backend = OpenAIBackend.__new__(OpenAIBackend)
        backend.model = "test"
        backend.json_mode = json_mode
        backend._mode = "schema" if json_mode in ("auto", "schema") else "prompt"
        backend._downgraded = False
        import threading
        backend._lock = threading.Lock()
        backend.tried = []

        def _call(system, user, schema, mode):
            backend.tried.append(mode)
            effect = effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect

        backend._call = _call
        return backend

    def choice(self, text):
        from extract.backends import Usage

        class Message:
            content = text
            reasoning_content = ""

        class Choice:
            message = Message()
            finish_reason = "stop"

        return Choice(), Usage(calls=1)

    def test_auto_retries_in_prompt_mode_after_a_rejection(self):
        backend = self.build([self.error(400), self.choice('{"invoice_number": "A1"}')])
        result = backend.complete("sys", "user", {})
        assert backend.tried == ["schema", "prompt"]
        assert not result.error
        assert result.mode == "prompt"

    def test_the_downgrade_sticks_for_the_rest_of_the_run(self):
        backend = self.build([self.error(400), self.choice('{"a": 1}')])
        backend.complete("sys", "user", {})
        assert backend._modes() == ["prompt"]

    def test_a_non_rejection_error_still_fails_the_document(self):
        """A timeout must not burn a second request to fail the same way."""
        backend = self.build([TimeoutError("timed out")])
        result = backend.complete("sys", "user", {})
        assert backend.tried == ["schema"]
        assert "TimeoutError" in result.error

    def test_schema_mode_does_not_fall_back(self):
        """An explicit `schema` is a demand, not a preference."""
        backend = self.build([self.error(400)], json_mode="schema")
        result = backend.complete("sys", "user", {})
        assert backend.tried == ["schema"]
        assert result.error
