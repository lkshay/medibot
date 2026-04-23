"""Shared finding/result types for input and output guards."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PhiFinding(BaseModel):
    entity_type: str = Field(..., description="e.g. PERSON, EMAIL_ADDRESS, PHONE_NUMBER, US_SSN")
    start: int
    end: int
    score: float = Field(..., description="Detector confidence in [0, 1]")
    original: str


class InjectionFinding(BaseModel):
    pattern: str
    matched_text: str
    severity: str = Field("medium", description="low | medium | high")


class InputGuardResult(BaseModel):
    """Returned by InputGuard.process(...)."""

    allowed: bool = Field(..., description="If False, the query should be rejected.")
    sanitized_text: str = Field(..., description="PHI-redacted user text to pass downstream.")
    phi_findings: list[PhiFinding] = Field(default_factory=list)
    injection_findings: list[InjectionFinding] = Field(default_factory=list)
    rejection_reason: str | None = None


class OutputGuardResult(BaseModel):
    """Returned by OutputGuard.process(...)."""

    final_answer: str
    emergency_injected: bool = False
    disclaimer_injected: bool = False
