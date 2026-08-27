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
            declared = {f.name for f in doctype.graded_fields()} | {g.name for g in doctype.groups}
            assert set(json_schema(doctype)["properties"]) == declared, name
            for variant in doctype.variants:
                expected = {f.name for f in doctype.graded_fields(variant)} |                            {g.name for g in doctype.groups}
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


class TestTruncationIsNotMalformedOutput:
    """A cut-off answer and a broken one need different messages.

    All twelve resume failures in the first full corpus run were the model still
    writing when it hit max_tokens. Every one was reported as "unparseable JSON",
    which reads like a broken schema and sends you to the wrong file. The fix is a
    bigger budget, so the error has to name the budget.
    """

    def result_for(self, text, truncated):
        from extract.backends import Completion, Usage
        from extract.runner import extract_document
        from core.doctypes import RESUME
        import os

        class Backend:
            def complete(self, system, user, schema):
                return Completion(text=text, usage=Usage(calls=1),
                                  truncated=truncated, mode="schema")

        sample = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tools", "document-generator", "samples", "invoice.pdf")
        if not os.path.exists(sample):
            pytest.skip("no sample document available")
        return extract_document(Backend(), RESUME, sample, "resumes/x.pdf")

    def test_a_cut_off_answer_reports_the_budget(self):
        result = self.result_for('{"name": "Ada Lovel', truncated=True)
        assert "truncated" in result.error
        assert "max_tokens" in result.error
        assert "unparseable" not in result.error

    def test_genuinely_malformed_output_still_says_so(self):
        result = self.result_for("Sure! Here is the JSON you asked for.", truncated=False)
        assert "unparseable JSON" in result.error
        assert "truncated" not in result.error

    def test_a_complete_answer_is_unaffected(self):
        result = self.result_for('{"name": "Ada Lovelace"}', truncated=False)
        assert not result.error
        assert result.record["name"] == "Ada Lovelace"


class TestOptionalFields:
    """Fields the document may not carry are asked as a decision, not a slot.

    A flat nullable field asks "what is the value?", and a required slot with no answer
    is pressure to invent one: 46 of 46 absent service locations were filled anyway,
    37 copied verbatim from a neighbour, and a model four times larger produced exactly
    the same count. Nullability was never the missing piece -- the type already allowed
    null and the description said so for three rewrites running.
    """

    def prop(self, variant="loan", name="co_applicant_name"):
        from core.doctypes import FORM
        return json_schema(FORM, variant)["properties"][name]

    def test_an_optional_field_asks_for_a_status(self):
        prop = self.prop()
        assert prop["type"] == "object"
        assert set(prop["properties"]) == {"status", "value"}
        assert prop["properties"]["status"]["enum"] == ["present", "absent", "unclear"]

    def test_the_value_keeps_the_type_it_would_have_had(self):
        from core.doctypes import REGISTRY
        sections = json_schema(REGISTRY["multi_bill_invoice"])["properties"]["sections"]
        prop = sections["items"]["properties"]["service_location"]
        assert prop["properties"]["value"]["type"] == ["string", "null"]

    def test_absence_guidance_reaches_the_model(self):
        assert "absent" in self.prop()["description"].lower()

    def test_a_required_field_is_left_alone(self):
        """Only fields that can legitimately be missing change shape."""
        assert json_schema(__import__("core.doctypes", fromlist=["FORM"]).FORM,
                           "loan")["properties"]["loan_amount"]["type"] == ["number", "null"]

    def test_optional_fields_are_still_required_keys(self):
        """Structured output needs every property listed; the object is not optional."""
        from core.doctypes import FORM
        schema = json_schema(FORM, "loan")
        assert "co_applicant_name" in schema["required"]
        assert set(self.prop()["required"]) == {"status", "value"}


class TestCollapseOptional:
    """The decision is flattened before anything downstream sees it.

    Rules, scorer and stored records all speak plain values. Teaching them a second
    shape would spread one extraction detail across the whole system.
    """

    def collapse(self, record, doctype=None, variant=""):
        from core.doctypes import REGISTRY
        from extract.runner import collapse_optional
        collapse_optional(record, doctype or REGISTRY["multi_bill_invoice"], variant)
        return record

    def test_present_keeps_the_value(self):
        rec = self.collapse({"sections": [
            {"service_location": {"status": "present", "value": "77 Oak Street"}}]})
        assert rec["sections"][0]["service_location"] == "77 Oak Street"

    def test_absent_becomes_none(self):
        rec = self.collapse({"sections": [
            {"service_location": {"status": "absent", "value": None}}]})
        assert rec["sections"][0]["service_location"] is None

    def test_unclear_becomes_none_and_discards_the_guess(self):
        """Optimised for precision: an invented address outlives a blank one."""
        rec = self.collapse({"sections": [
            {"service_location": {"status": "unclear", "value": "Meter M3947745"}}]})
        assert rec["sections"][0]["service_location"] is None

    def test_a_present_status_with_no_value_is_still_none(self):
        rec = self.collapse({"sections": [
            {"service_location": {"status": "present", "value": None}}]})
        assert rec["sections"][0]["service_location"] is None

    def test_top_level_optional_fields_collapse_too(self):
        from core.doctypes import FORM
        rec = self.collapse(
            {"co_applicant_name": {"status": "absent", "value": None},
             "loan_amount": 250000},
            doctype=FORM, variant="loan")
        assert rec["co_applicant_name"] is None
        assert rec["loan_amount"] == 250000      # untouched

    def test_non_optional_fields_are_never_touched(self):
        rec = self.collapse({"sections": [
            {"cost_center": "CC-2040 Operations", "service_code": "UTL-1"}]})
        assert rec["sections"][0]["cost_center"] == "CC-2040 Operations"

    def test_a_flat_answer_survives_a_backend_that_ignored_the_shape(self):
        """Prompt mode cannot be made to comply; a plain string must still work."""
        rec = self.collapse({"sections": [{"service_location": "77 Oak Street"}]})
        assert rec["sections"][0]["service_location"] == "77 Oak Street"

    def test_a_dict_without_a_status_is_left_as_is(self):
        rec = self.collapse({"sections": [{"service_location": {"value": "x"}}]})
        assert rec["sections"][0]["service_location"] == {"value": "x"}

    def test_it_survives_the_absences_the_record_contract_warns_about(self):
        assert self.collapse({}) == {}
        assert self.collapse({"sections": None}) == {"sections": None}
        assert self.collapse({"sections": ["not a dict"]}) == {"sections": ["not a dict"]}
        assert self.collapse({"sections": [{}]}) == {"sections": [{}]}


class TestAllFailedIsAFailure:
    """A run where nothing succeeded must not exit 0.

    Twice in one session a run wrote a predictions file, printed a summary and exited
    successfully having accomplished nothing -- once with the Docker daemon down, once
    when every request failed to reach the model server. Both looked like results until
    someone read the file. The second one wrote twelve identical connection errors.
    """

    def test_identical_failures_collapse_to_one_cause(self):
        from extract.cli import _cause
        a = _cause("APIConnectionError: Connection error.")
        b = _cause("APIConnectionError: Connection error.")
        assert a == b
        assert "APIConnectionError" in a

    def test_different_failures_stay_distinct(self):
        from extract.cli import _cause
        assert _cause("APIConnectionError: Connection error.") != \
               _cause("truncated: ran out of tokens mid-answer after 26198 characters")

    def test_document_specific_detail_does_not_split_one_cause(self):
        """Twelve documents failing the same way is one problem, not twelve."""
        from extract.cli import _cause
        assert _cause("unparseable JSON: Expecting value: line 1 column 25910") == \
               _cause("unparseable JSON: Expecting value: line 1 column 309")

    def test_a_message_with_no_colon_survives(self):
        from extract.cli import _cause
        assert _cause("something went wrong") == "something went wrong"


class TestRepeatingGroupsAreBounded:
    """An unbounded array in a constrained-decoding schema can loop forever.

    Under structured output the model is never invalid while emitting one more array
    element, so nothing forces it to close the array -- it can only stop by exhausting
    max_tokens. One resume did exactly that, repeating the same work-history entry
    until it had produced 49,853 characters where the normal prediction is 400-900.
    No timeout catches it either: a socket timeout fires on silence, and a looping
    model streams steadily.

    maxItems is a stop condition, not a claim about how many rows a document may have.
    The largest repeating group anywhere in the corpus is six line items.
    """

    def groups_in(self, schema):
        for name, prop in schema.get("properties", {}).items():
            if prop.get("type") == "array":
                yield name, prop
                yield from self.groups_in(prop.get("items", {}))

    def test_every_repeating_group_has_a_ceiling(self):
        from core.doctypes import REGISTRY
        for type_name, doctype in REGISTRY.items():
            found = list(self.groups_in(json_schema(doctype)))
            for name, prop in found:
                assert prop.get("maxItems"), f"{type_name}.{name} is unbounded"

    def test_nested_groups_are_bounded_too(self):
        """sections contain line_items; the inner array loops just as happily."""
        from core.doctypes import REGISTRY
        sections = json_schema(REGISTRY["multi_bill_invoice"])["properties"]["sections"]
        assert sections["maxItems"]
        assert sections["items"]["properties"]["line_items"]["maxItems"]

    def test_the_ceiling_clears_the_corpus_by_a_wide_margin(self):
        """A cap that truncates a real document would trade one bug for a worse one."""
        import glob
        import json as _json
        import os
        root = os.environ.get("DI_DATASET_ROOT", "/data")
        paths = glob.glob(os.path.join(root, "labels", "*.json"))
        if not paths:
            pytest.skip("no generated corpus available")

        def widest(rows):
            most = 0
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                for value in row.values():
                    if isinstance(value, list):
                        most = max(most, len(value), widest(value))
            return most

        observed = 0
        for path in paths:
            for record in _json.load(open(path, encoding="utf-8")):
                for group in ("line_items", "work_history", "sections"):
                    rows = record.get(group)
                    if isinstance(rows, list):
                        observed = max(observed, len(rows), widest(rows))
        from core.doctypes import REGISTRY
        ceiling = min(g.max_rows for d in REGISTRY.values() for g in d.groups)
        assert ceiling >= observed * 4, \
            f"ceiling {ceiling} is close to the corpus maximum {observed}; raise it"
