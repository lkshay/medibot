"""InputGuard — runs the pre-flight checks on every user query.

Order of operations:
    1. PHI/PII redaction — we never send raw PII to the LLM or Mem0.
    2. Prompt-injection detection — high-severity matches reject the query.

The returned `InputGuardResult.sanitized_text` is what downstream agents see.
The raw input is discarded.
"""

from __future__ import annotations

from .injection import detect_injection
from .phi import PhiRedactor
from .schemas import InputGuardResult


class InputGuard:
    def __init__(
        self,
        redact_phi: bool = True,
        block_injection: bool = True,
        redactor: PhiRedactor | None = None,
    ) -> None:
        self.redact_phi = redact_phi
        self.block_injection = block_injection
        self._redactor = redactor or PhiRedactor()

    def process(self, text: str) -> InputGuardResult:
        sanitized = text
        phi_findings = []

        # Run injection detection on the RAW text first — after Presidio redacts
        # names (e.g. "DAN" -> <PERSON>) we lose signal. Combine findings from
        # both raw and sanitized so we catch everything.
        raw_inj = detect_injection(text)

        if self.redact_phi:
            sanitized, phi_findings = self._redactor.redact(text)

        sanitized_inj = detect_injection(sanitized)
        # Merge uniquely by (pattern, matched_text)
        seen = {(f.pattern, f.matched_text) for f in raw_inj}
        injection_findings = list(raw_inj)
        for f in sanitized_inj:
            if (f.pattern, f.matched_text) not in seen:
                injection_findings.append(f)

        if self.block_injection and any(f.severity == "high" for f in injection_findings):
            return InputGuardResult(
                allowed=False,
                sanitized_text=sanitized,
                phi_findings=phi_findings,
                injection_findings=injection_findings,
                rejection_reason=(
                    "Your message matched a pattern commonly used in prompt-injection "
                    "attempts. Please rephrase your medical question without referring to "
                    "system prompts or role instructions."
                ),
            )

        return InputGuardResult(
            allowed=True,
            sanitized_text=sanitized,
            phi_findings=phi_findings,
            injection_findings=injection_findings,
        )
