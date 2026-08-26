"""Form variants, pinned against the corpus the generator actually produces.

The declaration in core/doctypes.py is hand-written; the corpus is generated. These
tests fail if the two drift, which is the only way to notice that a form type gained a
field and the extractor stopped asking for it.
"""
import json
import os

import pytest

from core.doctypes import FORM, NON_FIELD_KEYS, REGISTRY
from extract.schema import json_schema

CORPUS_LABELS = os.environ.get("DI_DATASET_ROOT", "/data") + "/labels/forms.json"


def corpus_fields_by_type():
    if not os.path.exists(CORPUS_LABELS):
        pytest.skip("no generated corpus available")
    records = json.load(open(CORPUS_LABELS, encoding="utf-8"))
    by_type = {}
    for record in records:
        keys = {k for k in record if k not in NON_FIELD_KEYS}
        by_type.setdefault(record["form_type"], set()).update(keys)
    return by_type


class TestVariantDeclaration:
    def test_every_corpus_form_type_is_declared(self):
        for form_type in corpus_fields_by_type():
            assert form_type in FORM.variants, f"{form_type} has no declared variant"

    @pytest.mark.parametrize("form_type", ["onboarding", "claim", "w9", "w4", "loan"])
    def test_variant_covers_every_field_the_corpus_carries(self, form_type):
        actual = corpus_fields_by_type().get(form_type)
        if actual is None:
            pytest.skip(f"no {form_type} documents in the corpus")
        declared = {f.name for f in FORM.fields_for(form_type)}
        missing = actual - declared
        assert not missing, f"{form_type} corpus has undeclared fields: {sorted(missing)}"

    def test_variant_does_not_ask_for_fields_the_type_never_has(self):
        """The whole point: a W-9 should not be asked for loan_amount."""
        by_type = corpus_fields_by_type()
        for form_type, actual in by_type.items():
            declared = {f.name for f in FORM.fields_for(form_type)}
            # form_type is shared, always present, and never extracted -- see
            # TestVariantKeyIsNotExtracted below.
            spurious = declared - actual - {"form_type"}
            assert not spurious, f"{form_type} is asked for absent fields: {sorted(spurious)}"


class TestNarrowing:
    def test_variant_schema_is_much_smaller_than_the_union(self):
        union = len(json_schema(FORM)["properties"])
        w9 = len(json_schema(FORM, "w9")["properties"])
        assert w9 < union / 3, f"w9 schema is {w9} of {union} fields; expected far fewer"

    def test_unknown_variant_falls_back_to_the_union(self):
        """Over-asking is wasteful; silently dropping real fields is wrong."""
        fallback = set(json_schema(FORM, "not-a-form-type")["properties"])
        union = set(json_schema(FORM)["properties"])
        assert fallback == union

    def test_variant_schema_still_satisfies_structured_output(self):
        for variant in FORM.variants:
            schema = json_schema(FORM, variant)
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])

    def test_types_without_variants_are_unaffected(self):
        for name, doctype in REGISTRY.items():
            if not doctype.variants:
                assert doctype.fields_for("anything") == doctype.fields, name

    def test_prompt_names_the_variant(self):
        from extract.schema import instructions
        assert "w9" in instructions(FORM, "w9").lower()


class TestSectionIdentifiers:
    """The multi-bill identifiers must be distinguishable, or the model rotates them."""

    def test_each_section_identifier_has_its_own_description(self):
        sections = json_schema(REGISTRY["multi_bill_invoice"])["properties"]["sections"]
        props = sections["items"]["properties"]
        descriptions = {
            name: props[name].get("description", "")
            for name in ("service_code", "account_number", "reference_number")
        }
        assert all(descriptions.values()), "an identifier has no description"
        assert len(set(descriptions.values())) == 3, \
            "identifiers share a description; the model cannot tell them apart"

    def test_reference_label_is_not_extracted(self):
        """It is the name of the value beside it, not a value.

        The page prints `METER M3947745`. Asked for a "reference label" the model
        returns the whole phrase, because on the document that word is the label for
        the number -- there is no separate value to transcribe. It stays in the corpus
        as ground truth about the document and out of the extraction schema.
        """
        props = json_schema(REGISTRY["multi_bill_invoice"])["properties"]
        section = props["sections"]["items"]["properties"]
        assert "reference_label" not in section
        # The number it labels is still asked for, and still says what it is.
        assert "meter" in section["reference_number"]["description"].lower()


class TestVariantKeyIsNotExtracted:
    """The key that selected the schema is not a field the schema asks for.

    Asked for a `form_type`, a model returns what the form calls itself on the page --
    "HR-ONB-1002", "New Hire Onboarding Form" -- because "onboarding" is a name in our
    taxonomy and appears nowhere on the document. Every one of the 160 forms in the
    first baseline was wrong on this single field, and none of them was the model's
    fault. It is the same defect as asking for `reference_label`.
    """

    def test_it_is_absent_from_every_variant_schema(self):
        for variant in FORM.variants:
            assert "form_type" not in json_schema(FORM, variant)["properties"], variant

    def test_it_is_absent_from_the_union_schema(self):
        """Deciding it from the document is classification, not extraction."""
        assert "form_type" not in json_schema(FORM)["properties"]

    def test_the_rest_of_the_variant_survives(self):
        w9 = json_schema(FORM, "w9")["properties"]
        assert "tax_classification" in w9 and "ssn" in w9

    def test_types_without_a_variant_key_lose_nothing(self):
        for name, doctype in REGISTRY.items():
            if doctype.variant_key:
                continue
            declared = {f.name for f in doctype.fields}
            assert set(json_schema(doctype)["properties"]) >= declared, name

    def test_the_schema_is_still_valid_for_structured_output(self):
        for variant in FORM.variants:
            schema = json_schema(FORM, variant)
            assert set(schema["required"]) == set(schema["properties"])
            assert schema["additionalProperties"] is False

    def test_the_runner_keeps_the_corpus_value(self):
        """A prompt-mode backend can return anything; update() must not let it win."""
        from extract.runner import extract_document

        class Backend:
            def complete(self, system, user, schema):
                from extract.backends import Completion, Usage
                return Completion(text='{"form_type": "HR-ONB-1002", "ssn": "1"}',
                                  usage=Usage(calls=1), mode="prompt")

        import os
        sample = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "tools", "document-generator", "samples", "form-w9.pdf")
        if not os.path.exists(sample):
            pytest.skip("no sample document available")
        result = extract_document(Backend(), FORM, sample, "forms/x.pdf", variant="w9")
        assert result.record["form_type"] == "w9"
