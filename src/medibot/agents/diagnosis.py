"""Diagnosis agent — FAISS-backed disease retrieval from a free-text query.

Pipeline:
    user query  ->  embed  ->  FAISS top-k  ->  enrich with symptom overlap
                                              ->  DiagnosisOutput

The agent does NOT commit to a diagnosis; it surfaces ranked candidates.
The orchestrator (M4) decides whether to ask the user for more symptoms
(when top-1 and top-2 are within a small margin) or to proceed.
"""

from __future__ import annotations

from ..data import MedicalData, normalize_text
from ..observability import get_tracer
from ..rag.vector_store import FaissStore
from .schemas import DiagnosisInput, DiagnosisOutput, DiseaseCandidate


CONFIDENCE_GAP = 0.05


class DiagnosisAgent:
    """Lightweight wrapper around FaissStore + MedicalData for the agent interface."""

    name = "diagnosis"
    description = (
        "Given a free-text description of symptoms, return the top-k most likely "
        "diseases from the medical knowledge base. Use for any query that asks "
        "'what might I have' or describes a set of symptoms."
    )

    def __init__(self, store: FaissStore, data: MedicalData) -> None:
        self.store = store
        self.data = data

    def _matched_symptoms(self, disease: str, query: str) -> list[str]:
        """Which of this disease's canonical symptoms appear (by token) in the query?"""
        q_norm = " ".join((normalize_text(query) or "").split())
        # Check each token variant of every symptom (both spaced and underscored)
        hits: list[str] = []
        for sym in self.data.disease_symptoms.get(disease, set()):
            readable = sym.replace("_", " ")
            if sym in q_norm or readable in q_norm:
                hits.append(sym)
        return sorted(hits)

    def __call__(self, inp: DiagnosisInput) -> DiagnosisOutput:
        tracer = get_tracer()
        with tracer.span(
            name="faiss.search",
            as_type="retriever",
            input={"query": inp.query, "k": inp.k},
        ) as rspan:
            hits = self.store.search(inp.query, k=inp.k)
            rspan.update(
                output={
                    "candidates": [
                        {"disease": h.disease, "score": round(h.score, 4)} for h in hits
                    ]
                }
            )

        candidates = [
            DiseaseCandidate(
                disease=h.disease,
                score=round(h.score, 4),
                matched_symptoms=self._matched_symptoms(h.disease, inp.query),
            )
            for h in hits
        ]
        confident = (
            len(candidates) >= 2
            and (candidates[0].score - candidates[1].score) >= CONFIDENCE_GAP
        )
        return DiagnosisOutput(
            query=inp.query,
            candidates=candidates,
            confident=confident,
        )
