"""Unit tests for the 4 specialized agents (no LLM required)."""

import pytest

from medibot.agents import (
    DescriptionAgent,
    DescriptionInput,
    PrecautionAgent,
    PrecautionInput,
    SeverityAgent,
    SeverityInput,
    bucket_urgency,
)
from medibot.agents.severity import _detect_emergency_phrases, _fuzzy_match
from medibot.data import load_medical_data


@pytest.fixture(scope="module")
def data():
    return load_medical_data("data/raw")


class TestBucketUrgency:
    @pytest.mark.parametrize("total,expected", [
        (0, "low"),
        (5, "low"),
        (6, "moderate"),
        (10, "moderate"),
        (11, "urgent"),
        (15, "urgent"),
        (16, "emergency"),
        (30, "emergency"),
    ])
    def test_thresholds(self, total, expected):
        assert bucket_urgency(total) == expected


class TestFuzzyMatch:
    def test_exact_match(self, data):
        match = _fuzzy_match("itching", data.symptom_severity)
        assert match == ("itching", 1)

    def test_underscore_variant(self, data):
        match = _fuzzy_match("chest pain", data.symptom_severity)
        assert match == ("chest_pain", 7)

    def test_token_overlap(self, data):
        # "skin rash with patches" should match skin_rash
        match = _fuzzy_match("skin rash", data.symptom_severity)
        assert match is not None
        assert match[0] == "skin_rash"

    def test_no_match(self, data):
        assert _fuzzy_match("completely_unknown_symptom_xyz", data.symptom_severity) is None


class TestEmergencyPhraseDetection:
    @pytest.mark.parametrize("text,expected", [
        ("sudden sharp chest pain", "chest_pain"),
        ("crushing chest pressure", "chest_pain"),
        ("trouble breathing", "breathlessness"),
        ("shortness of breath", "breathlessness"),
        ("left arm weakness", "weakness_of_one_body_side"),
        ("can't move my left side", "weakness_of_one_body_side"),
        ("face drooping on one side", "weakness_of_one_body_side"),
        ("I feel confused and disoriented", "altered_sensorium"),
    ])
    def test_matches(self, text, expected):
        hits = _detect_emergency_phrases(text)
        assert expected in hits, f"expected {expected} in hits for {text!r}, got {hits}"

    @pytest.mark.parametrize("text", [
        "I have a mild headache",
        "itching and skin rash",
        "fever and cough",
    ])
    def test_non_emergency_doesnt_match(self, text):
        assert _detect_emergency_phrases(text) == []


class TestSeverityAgent:
    def test_low_urgency_mild_symptoms(self, data):
        sev = SeverityAgent(data)
        out = sev(SeverityInput(symptoms=["itching"]))
        assert out.urgency == "low"
        assert out.emergency_flags == []

    def test_chest_pain_triggers_emergency(self, data):
        sev = SeverityAgent(data)
        out = sev(SeverityInput(symptoms=["chest_pain"]))
        assert out.urgency == "emergency"
        assert "chest_pain" in out.emergency_flags

    def test_chest_pain_non_canonical_triggers_emergency(self, data):
        sev = SeverityAgent(data)
        out = sev(SeverityInput(symptoms=["sudden chest pain", "trouble breathing"]))
        assert out.urgency == "emergency"

    def test_stroke_phrase_triggers_emergency(self, data):
        sev = SeverityAgent(data)
        out = sev(SeverityInput(
            symptoms=["left arm weakness", "face drooping"]
        ))
        assert out.urgency == "emergency"
        assert "weakness_of_one_body_side" in out.emergency_flags

    def test_unrecognized_reported(self, data):
        sev = SeverityAgent(data)
        out = sev(SeverityInput(symptoms=["itching", "unicorn_horn"]))
        assert "unicorn_horn" in out.unrecognized


class TestDescriptionAgent:
    def test_lookup(self, data):
        agent = DescriptionAgent(data)
        out = agent(DescriptionInput(disease="diabetes"))
        assert out.found is True
        assert len(out.description) > 0

    def test_alias_resolution(self, data):
        # User typed the alternate spelling
        agent = DescriptionAgent(data)
        out = agent(DescriptionInput(disease="Dimorphic hemorrhoids(piles)"))
        assert out.found is True
        assert out.disease == "dimorphic hemmorhoids(piles)"

    def test_unknown_disease(self, data):
        agent = DescriptionAgent(data)
        out = agent(DescriptionInput(disease="not a real disease"))
        assert out.found is False


class TestPrecautionAgent:
    def test_lookup(self, data):
        agent = PrecautionAgent(data)
        out = agent(PrecautionInput(disease="chicken pox"))
        assert out.found is True
        assert len(out.precautions) > 0

    def test_alias(self, data):
        agent = PrecautionAgent(data)
        out = agent(PrecautionInput(disease="Dimorphic hemorrhoids(piles)"))
        assert out.found is True
        assert len(out.precautions) == 4
