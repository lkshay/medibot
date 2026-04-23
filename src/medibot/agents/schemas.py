"""Pydantic I/O schemas for the four specialized agent tools.

Why Pydantic:
- Runtime validation — catches bad arguments before they reach the tool
- Auto-generated JSON schema — the orchestrator (M4) will advertise these
  shapes to the LLM so it produces well-formed tool calls
- IDE / typechecker support
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ============================================================================
# Diagnosis
# ============================================================================
class DiagnosisInput(BaseModel):
    query: str = Field(
        ...,
        description="Free-text user description of symptoms, e.g. 'I have itching and a red rash'",
    )
    k: int = Field(3, ge=1, le=10, description="Number of candidate diseases to return")


class DiseaseCandidate(BaseModel):
    disease: str
    score: float = Field(..., description="Cosine similarity to the query (higher is better)")
    matched_symptoms: list[str] = Field(
        default_factory=list,
        description="Subset of this disease's known symptoms that textually appear in the query",
    )


class DiagnosisOutput(BaseModel):
    query: str
    candidates: list[DiseaseCandidate]
    confident: bool = Field(
        ...,
        description="True if top-1 score is clearly above the runner-up (gap >= 0.05)",
    )


# ============================================================================
# Severity
# ============================================================================
class SeverityInput(BaseModel):
    symptoms: list[str] = Field(
        ...,
        description="Symptom tokens (e.g. 'skin_rash', 'high_fever')",
    )


class SeverityOutput(BaseModel):
    recognized: dict[str, int] = Field(
        default_factory=dict, description="Symptom -> severity weight (1-7)"
    )
    unrecognized: list[str] = Field(
        default_factory=list, description="Symptoms not found in the severity table"
    )
    total_weight: int
    urgency: str = Field(
        ..., description="One of: low, moderate, urgent, emergency"
    )
    emergency_flags: list[str] = Field(
        default_factory=list,
        description="Symptoms that by themselves trigger an emergency escalation",
    )
    note: str


# ============================================================================
# Description
# ============================================================================
class DescriptionInput(BaseModel):
    disease: str


class DescriptionOutput(BaseModel):
    disease: str
    description: str
    found: bool


# ============================================================================
# Precaution
# ============================================================================
class PrecautionInput(BaseModel):
    disease: str


class PrecautionOutput(BaseModel):
    disease: str
    precautions: list[str]
    found: bool
