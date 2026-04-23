"""Severity agent — no LLM needed; it's a lookup + thresholds.

Responsibilities:
1. Look up the severity weight (1-7) of each user-reported symptom.
2. Sum the weights; bucket into urgency: low | moderate | urgent | emergency.
3. Flag hard-coded emergency symptoms that override the sum (e.g., chest pain
   should escalate regardless of total weight).

Symptom matching is fuzzy — the LLM often passes variations like "chest pain"
when the canonical form is "chest_pain", or "left-side weakness" for
"weakness_of_one_body_side". Pure exact-match was the #1 cause of severity
escalation being missed in evals, so we fall back to token-overlap matching.

The thresholds below are deliberately conservative for a medical-adjacent
demo. Production systems would validate these with a clinician.
"""

from __future__ import annotations

import re

from ..data import MedicalData, normalize_text
from .schemas import SeverityInput, SeverityOutput


# Symptoms that by themselves warrant emergency escalation, regardless of sum.
# Names use the canonical (lower, underscore-joined) form used in the corpus.
_EMERGENCY_SYMPTOMS: set[str] = {
    "chest_pain",
    "breathlessness",
    "altered_sensorium",
    "coma",
    "stomach_bleeding",
    "acute_liver_failure",
    "weakness_of_one_body_side",
    "loss_of_smell",  # combined with fever can indicate emergent infection; conservative
}


# Natural-language phrases that map to canonical emergency symptoms. We check
# these against each raw input *before* fuzzy matching so LLM-paraphrased
# symptoms still escalate correctly. A stroke described as "I can't move my
# left arm" would otherwise match only `muscle_weakness` (weight 2).
_EMERGENCY_PHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bchest\s+(?:pain|hurts?|pressure|tight(?:ness)?|crushing)", re.I), "chest_pain"),
    (re.compile(r"\bcrushing\s+chest", re.I), "chest_pain"),
    (re.compile(r"\b(?:difficulty|trouble|can'?t|cannot)\s+breath", re.I), "breathlessness"),
    (re.compile(r"\bshort(?:ness)?\s+of\s+breath", re.I), "breathlessness"),
    (re.compile(r"\bgasp(?:ing)?\b", re.I), "breathlessness"),
    (re.compile(r"\bone[\s-]sided\s+(?:weakness|numbness|paralys)", re.I), "weakness_of_one_body_side"),
    (re.compile(r"\b(?:left|right)\s+(?:arm|leg|side|face|body|hand|foot)\s+(?:weak|numb|paralyz|droop|can'?t\s+move)", re.I), "weakness_of_one_body_side"),
    (re.compile(r"\b(?:can'?t|cannot)\s+move\s+(?:my|the)\s+(?:left|right)", re.I), "weakness_of_one_body_side"),
    (re.compile(r"\bface\s+(?:droop|numbness|weakness|numb)", re.I), "weakness_of_one_body_side"),
    (re.compile(r"\bhalf\s+(?:of\s+)?(?:my\s+)?face", re.I), "weakness_of_one_body_side"),
    (re.compile(r"\bstroke\b", re.I), "weakness_of_one_body_side"),
    (re.compile(r"\bconfus(?:ed|ion)|disorient(?:ed|ation)|unconscious|passed\s+out|blacking\s+out\b", re.I), "altered_sensorium"),
    (re.compile(r"\bcoma\b", re.I), "coma"),
    (re.compile(r"\bblood(?:y)?\s+(?:stool|vomit)", re.I), "stomach_bleeding"),
    (re.compile(r"\bvomit(?:ing)?\s+blood", re.I), "stomach_bleeding"),
]


def _detect_emergency_phrases(text: str) -> list[str]:
    """Return canonical emergency-symptom names that match a given input string."""
    hits: list[str] = []
    for pat, canonical in _EMERGENCY_PHRASES:
        if pat.search(text):
            if canonical not in hits:
                hits.append(canonical)
    return hits


def bucket_urgency(total: int) -> str:
    """Translate total severity weight to an urgency level."""
    if total >= 16:
        return "emergency"
    if total >= 11:
        return "urgent"
    if total >= 6:
        return "moderate"
    return "low"


def _fuzzy_match(raw: str, known: dict[str, int]) -> tuple[str, int] | None:
    """Best-effort canonical lookup. Returns (canonical, weight) or None.

    Order:
        1. Exact normalized match
        2. Underscore-joined variant
        3. Token-overlap heuristic: share at least 1 token and cover ≥ 50% of
           the canonical's tokens.
    """
    n = normalize_text(raw) or ""
    if not n:
        return None
    if n in known:
        return (n, known[n])
    alt = n.replace(" ", "_")
    if alt in known:
        return (alt, known[alt])

    q_tokens = set(n.replace("_", " ").split()) - {"of", "and", "the", "a", "an", "my"}
    if not q_tokens:
        return None

    best: tuple[str, int] | None = None
    best_ratio = 0.0
    for canonical, weight in known.items():
        can_tokens = set(canonical.replace("_", " ").split()) - {"of", "and", "the"}
        if not can_tokens:
            continue
        overlap = q_tokens & can_tokens
        if not overlap:
            continue
        # Require the canonical's own tokens are well covered by the query.
        ratio = len(overlap) / len(can_tokens)
        if ratio >= 0.5 and ratio > best_ratio:
            best = (canonical, weight)
            best_ratio = ratio
    return best


class SeverityAgent:
    name = "severity"
    description = (
        "Given a list of symptoms, return their combined severity score and an "
        "urgency level ('low', 'moderate', 'urgent', 'emergency'). Use when the "
        "user asks 'how bad is this' or when you need to decide whether to "
        "recommend a doctor visit."
    )

    def __init__(self, data: MedicalData) -> None:
        self.data = data

    def __call__(self, inp: SeverityInput) -> SeverityOutput:
        recognized: dict[str, int] = {}
        unrecognized: list[str] = []
        emergency_flags: list[str] = []

        for raw in inp.symptoms:
            # Emergency-phrase check FIRST — paraphrased emergencies
            # ("left arm weakness" → stroke; "can't breathe" → breathlessness)
            # are the safety-critical misses we most want to catch.
            phrase_hits = _detect_emergency_phrases(raw)
            for canonical in phrase_hits:
                if canonical not in emergency_flags:
                    emergency_flags.append(canonical)
                if canonical not in recognized:
                    recognized[canonical] = int(
                        self.data.symptom_severity.get(canonical, 7)
                    )
            if phrase_hits:
                # phrase matched this symptom; no need to also fuzzy-match it
                continue

            match = _fuzzy_match(raw, self.data.symptom_severity)
            if match is None:
                unrecognized.append(raw)
                continue
            canonical, weight = match
            recognized[canonical] = int(weight)
            if canonical in _EMERGENCY_SYMPTOMS:
                emergency_flags.append(canonical)

        total = sum(recognized.values())
        urgency = "emergency" if emergency_flags else bucket_urgency(total)

        if urgency == "emergency":
            note = (
                "One or more symptoms indicate a potential emergency. Seek "
                "immediate medical attention or call emergency services."
            )
        elif urgency == "urgent":
            note = "Combined severity is high. Recommend consulting a doctor soon."
        elif urgency == "moderate":
            note = "Combined severity is moderate. Monitor and consider a consultation."
        else:
            note = "Combined severity is low. Self-care and observation are reasonable."

        return SeverityOutput(
            recognized=recognized,
            unrecognized=unrecognized,
            total_weight=total,
            urgency=urgency,
            emergency_flags=emergency_flags,
            note=note,
        )
