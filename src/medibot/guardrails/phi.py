"""PHI / PII redaction via Microsoft Presidio.

Presidio is a two-engine design:
    AnalyzerEngine    — detects entities (names, SSNs, emails, etc.) using
                        spaCy NER + pattern recognizers.
    AnonymizerEngine  — replaces / masks detected spans in the original text.

We wrap both in a single class with a simple .redact() method. Presidio is
heavy to import, so we lazy-load on first use.
"""

from __future__ import annotations

from typing import Optional

from .schemas import PhiFinding


# Default entity set tuned for a medical chat bot.
# Deliberately EXCLUDES DATE_TIME and LOCATION — in medical Q&A, symptom
# duration ("2 days") and travel history ("Paris last week") are diagnostic
# signal, not identifiers. A hospital EHR pipeline would include these; a
# symptom checker shouldn't.
_DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "URL",
    "MEDICAL_LICENSE",
    "US_DRIVER_LICENSE",
]


class PhiRedactor:
    """Detects and masks PHI/PII in free-text user input."""

    def __init__(self, entities: Optional[list[str]] = None, threshold: float = 0.4) -> None:
        self._entities = entities or _DEFAULT_ENTITIES
        self._threshold = threshold
        self._analyzer = None
        self._anonymizer = None

    def _ensure_loaded(self) -> None:
        if self._analyzer is None:
            # Lazy import — Presidio pulls in spaCy and is ~1s to initialize.
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()

    def redact(self, text: str) -> tuple[str, list[PhiFinding]]:
        """Return (redacted_text, findings)."""
        if not text or not text.strip():
            return text, []
        self._ensure_loaded()
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self._entities,
            score_threshold=self._threshold,
        )
        if not results:
            return text, []
        findings = [
            PhiFinding(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                score=float(r.score),
                original=text[r.start : r.end],
            )
            for r in results
        ]
        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, findings
