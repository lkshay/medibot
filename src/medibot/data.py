"""Canonical loader for the MediBot medical dataset.

All downstream components (EDA notebook, RAG index, agents) import from here
so there is a single source of truth for data cleaning, normalization, and
the cross-CSV disease-name alias fixups that EDA uncovered.

Key exports:
    normalize_text     — whitespace/case normalization for symptoms & queries
    normalize_disease  — same, for disease names, applying the alias table
    MedicalData        — dataclass bundling all 4 CSVs into clean dicts
    load_medical_data  — entry point
    disease_to_document — turn one disease into a rich text doc for embedding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


# ----------------------------------------------------------------------------
# Cross-CSV name fixups discovered during EDA.
# Keys are the *alias* (wrong spelling); values are the *canonical* name
# (the spelling used in dataset.csv, which is the source of truth for
# disease enumeration because it drives the diagnosis signal).
# ----------------------------------------------------------------------------
_DISEASE_ALIASES: dict[str, str] = {
    "dimorphic hemorrhoids(piles)": "dimorphic hemmorhoids(piles)",
}


def normalize_text(s: str | float | None) -> str | None:
    """Trim, lowercase, and collapse interior whitespace. Returns None for NaN/empty."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    cleaned = " ".join(str(s).strip().lower().split())
    return cleaned or None


def normalize_disease(s: str | float | None) -> str | None:
    """Disease-name normalization: text normalization + alias resolution."""
    n = normalize_text(s)
    if n is None:
        return None
    return _DISEASE_ALIASES.get(n, n)


@dataclass
class MedicalData:
    """Clean, normalized, in-memory representation of the MediBot corpus."""

    diseases: list[str]                             # canonical disease names
    disease_symptoms: dict[str, set[str]]           # disease -> set of symptoms
    symptom_severity: dict[str, int]                # symptom -> weight (1-7)
    disease_description: dict[str, str]             # disease -> long description
    disease_precautions: dict[str, list[str]]       # disease -> ordered precautions

    def symptoms_of(self, disease: str) -> set[str]:
        return self.disease_symptoms.get(normalize_disease(disease) or "", set())

    def severity_of(self, symptom: str) -> int | None:
        return self.symptom_severity.get(normalize_text(symptom) or "")

    def severity_sum(self, symptoms: Iterable[str]) -> int:
        """Sum of severity weights for any symptoms we can match; unknowns are skipped."""
        return sum(self.severity_of(s) or 0 for s in symptoms)


def load_medical_data(data_dir: Path | str = "data/raw") -> MedicalData:
    """Load and normalize all 4 CSVs, returning a MedicalData bundle."""
    data_dir = Path(data_dir)
    df_map = pd.read_csv(data_dir / "dataset.csv")
    df_sev = pd.read_csv(data_dir / "Symptom-severity.csv")
    df_desc = pd.read_csv(data_dir / "symptom_Description.csv")
    df_prec = pd.read_csv(data_dir / "symptom_precaution.csv")

    # ---- dataset.csv (disease -> 17 symptom columns) --------------------------
    symptom_cols = [c for c in df_map.columns if c.startswith("Symptom_")]
    df_map["Disease"] = df_map["Disease"].map(normalize_disease)
    for c in symptom_cols:
        df_map[c] = df_map[c].map(normalize_text)

    disease_symptoms: dict[str, set[str]] = {}
    for _, row in df_map.iterrows():
        disease = row["Disease"]
        if not disease:
            continue
        bucket = disease_symptoms.setdefault(disease, set())
        for c in symptom_cols:
            sym = row[c]
            # pandas can restore None to NaN (float) in object columns — guard against it
            if pd.notna(sym) and isinstance(sym, str) and sym:
                bucket.add(sym)

    # ---- severity -------------------------------------------------------------
    df_sev.columns = [c.strip() for c in df_sev.columns]
    df_sev["Symptom"] = df_sev["Symptom"].map(normalize_text)
    df_sev = df_sev.dropna(subset=["Symptom"])
    symptom_severity = dict(zip(df_sev["Symptom"], df_sev["weight"].astype(int)))

    # ---- description ----------------------------------------------------------
    df_desc["Disease"] = df_desc["Disease"].map(normalize_disease)
    df_desc = df_desc.dropna(subset=["Disease"])
    disease_description = {
        row["Disease"]: str(row["Description"]).strip()
        for _, row in df_desc.iterrows()
    }

    # ---- precautions ----------------------------------------------------------
    df_prec["Disease"] = df_prec["Disease"].map(normalize_disease)
    df_prec = df_prec.dropna(subset=["Disease"])
    prec_cols = [c for c in df_prec.columns if c.startswith("Precaution_")]
    disease_precautions: dict[str, list[str]] = {}
    for _, row in df_prec.iterrows():
        items = [str(row[c]).strip() for c in prec_cols
                 if pd.notna(row[c]) and str(row[c]).strip()]
        disease_precautions[row["Disease"]] = items

    diseases = sorted(disease_symptoms.keys())

    # ---- coverage check (loud failure, not silent) ---------------------------
    missing_desc = [d for d in diseases if d not in disease_description]
    missing_prec = [d for d in diseases if d not in disease_precautions]
    if missing_desc:
        raise ValueError(f"Diseases without descriptions: {missing_desc}")
    if missing_prec:
        raise ValueError(f"Diseases without precautions: {missing_prec}")

    return MedicalData(
        diseases=diseases,
        disease_symptoms=disease_symptoms,
        symptom_severity=symptom_severity,
        disease_description=disease_description,
        disease_precautions=disease_precautions,
    )


def disease_to_document(disease: str, data: MedicalData) -> str:
    """Render one disease as a rich text document suitable for embedding.

    Strategy: embed the disease's identity, its symptom profile, and a brief
    description in a single blob. Symptoms are repeated as bare tokens *and*
    in a readable sentence so lexical overlap with user queries stays high
    without sacrificing semantic richness.
    """
    symptoms = sorted(data.disease_symptoms[disease])
    symptoms_readable = ", ".join(s.replace("_", " ") for s in symptoms)
    description = data.disease_description.get(disease, "").strip()

    return (
        f"Disease: {disease}.\n"
        f"Common symptoms: {symptoms_readable}.\n"
        f"Description: {description}\n"
        f"Symptom tokens: {' '.join(symptoms)}"
    )
