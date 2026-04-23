"""Description agent — returns the plain-language description for a disease."""

from __future__ import annotations

from ..data import MedicalData, normalize_disease
from .schemas import DescriptionInput, DescriptionOutput


class DescriptionAgent:
    name = "description"
    description = (
        "Given a disease name, return its plain-language description. Use after "
        "the diagnosis agent has identified a likely disease, or whenever the "
        "user asks 'what is X' about a medical condition."
    )

    def __init__(self, data: MedicalData) -> None:
        self.data = data

    def __call__(self, inp: DescriptionInput) -> DescriptionOutput:
        canonical = normalize_disease(inp.disease) or inp.disease
        desc = self.data.disease_description.get(canonical)
        return DescriptionOutput(
            disease=canonical,
            description=desc or "",
            found=desc is not None,
        )
