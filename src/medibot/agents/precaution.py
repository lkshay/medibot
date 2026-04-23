"""Precaution agent — returns preventive / self-care actions for a disease."""

from __future__ import annotations

from ..data import MedicalData, normalize_disease
from .schemas import PrecautionInput, PrecautionOutput


class PrecautionAgent:
    name = "precaution"
    description = (
        "Given a disease name, return a list of preventive measures and self-care "
        "actions. Use after the disease is identified, or when the user asks "
        "'what should I do about X'."
    )

    def __init__(self, data: MedicalData) -> None:
        self.data = data

    def __call__(self, inp: PrecautionInput) -> PrecautionOutput:
        canonical = normalize_disease(inp.disease) or inp.disease
        items = self.data.disease_precautions.get(canonical, [])
        return PrecautionOutput(
            disease=canonical,
            precautions=items,
            found=bool(items),
        )
