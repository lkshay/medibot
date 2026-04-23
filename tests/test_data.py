"""Unit tests for the data loader and normalization helpers."""

from medibot.data import (
    disease_to_document,
    load_medical_data,
    normalize_disease,
    normalize_text,
)


class TestNormalize:
    def test_strip_and_lowercase(self):
        assert normalize_text("  Skin_Rash  ") == "skin_rash"

    def test_collapse_interior_whitespace(self):
        assert normalize_text("high   fever") == "high fever"

    def test_none_and_empty(self):
        assert normalize_text(None) is None
        assert normalize_text("") is None
        assert normalize_text("   ") is None

    def test_nan_float(self):
        import math
        assert normalize_text(math.nan) is None


class TestDiseaseAlias:
    def test_hemorrhoids_alias(self):
        # EDA uncovered: "Dimorphic hemorrhoids" in desc CSV maps to
        # "Dimorphic hemmorhoids" (double m) in dataset.csv — canonical.
        assert normalize_disease("Dimorphic hemorrhoids(piles)") == "dimorphic hemmorhoids(piles)"

    def test_pass_through(self):
        assert normalize_disease("Fungal Infection") == "fungal infection"


class TestLoadMedicalData:
    def setup_method(self):
        self.data = load_medical_data("data/raw")

    def test_expected_counts(self):
        assert len(self.data.diseases) == 41
        assert len(self.data.disease_description) == 41
        assert len(self.data.disease_precautions) == 41
        # 133 severity entries in the source CSV; one may collapse after dedup
        assert 130 <= len(self.data.symptom_severity) <= 133

    def test_all_diseases_have_description_and_precautions(self):
        """Coverage check — any gap means silent answer failures downstream."""
        for d in self.data.diseases:
            assert d in self.data.disease_description, d
            assert d in self.data.disease_precautions, d
            assert len(self.data.disease_precautions[d]) > 0, d

    def test_hemorrhoids_alias_resolved(self):
        assert "dimorphic hemmorhoids(piles)" in self.data.disease_description
        assert "dimorphic hemmorhoids(piles)" in self.data.disease_precautions

    def test_symptoms_are_normalized(self):
        for sym_set in self.data.disease_symptoms.values():
            for s in sym_set:
                assert s == normalize_text(s), s  # idempotent

    def test_severity_lookup(self):
        # Known canonical weights from the source data
        assert self.data.severity_of("itching") == 1
        assert self.data.severity_of("chest_pain") == 7

    def test_severity_sum_skips_unknown(self):
        total = self.data.severity_sum(["itching", "chest_pain", "not_a_symptom"])
        assert total == 8


class TestDiseaseToDocument:
    def test_document_has_name_and_description(self):
        data = load_medical_data("data/raw")
        doc = disease_to_document("diabetes", data)
        assert "diabetes" in doc.lower()
        assert len(doc) > 100
        # symptoms appear both readable and tokenized
        assert "_" in doc  # token form preserved for lexical matching
