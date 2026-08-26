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
            # form_type itself is shared and always present
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
