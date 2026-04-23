"""OutputGuard — enforces policy on the model's final answer.

Two invariants we enforce regardless of what the LLM says:

    1. If ANY tool observation reported urgency="emergency", the final answer
       MUST lead with an emergency banner. If the LLM forgets (rare, but
       possible), we prepend one.

    2. Every final answer must contain the medical disclaimer. We check and
       append if missing.

These are belt-and-suspenders over the prompt-level rules. Prompts guide,
guards enforce.
"""

from __future__ import annotations

import re

from ..orchestrator import ReactRun
from .schemas import OutputGuardResult


_EMERGENCY_BANNER = (
    "⚠️ **EMERGENCY: Your symptoms may require immediate medical attention. "
    "Please call your local emergency services or go to the nearest emergency "
    "room right now.**\n\n"
)

_DISCLAIMER = (
    "\n\n*This is informational only and not a medical diagnosis; please "
    "consult a licensed clinician for a definitive assessment.*"
)

_EMERGENCY_HINT_RE = re.compile(
    r"(?i)\b(?:emergency|urgent|immediate(?:ly)?|seek\s+immediate|call\s+9\s*1\s*1|"
    r"ambulance|right\s+away|hospital\s+now|go\s+to\s+the\s+(?:ER|emergency))\b|⚠️|🚨"
)
_DISCLAIMER_HINTS = ("clinician", "licensed", "medical professional", "informational only", "not a medical diagnosis")
_EMERGENCY_OBSERVATION = re.compile(r'"urgency"\s*:\s*"emergency"')


class OutputGuard:
    def process(self, answer: str, run: ReactRun) -> OutputGuardResult:
        out = answer or ""
        out = out.strip()

        # 1. Emergency override — did any tool say this was an emergency?
        emergency = any(
            step.observation and _EMERGENCY_OBSERVATION.search(step.observation)
            for step in run.steps
        )
        emergency_injected = False
        if emergency:
            leader = out[:240]
            if not _EMERGENCY_HINT_RE.search(leader):
                out = _EMERGENCY_BANNER + out
                emergency_injected = True

        # 2. Disclaimer enforcement
        disclaimer_injected = False
        tail = out[-300:].lower()
        if not any(tok in tail for tok in (h.lower() for h in _DISCLAIMER_HINTS)):
            out = out + _DISCLAIMER
            disclaimer_injected = True

        return OutputGuardResult(
            final_answer=out,
            emergency_injected=emergency_injected,
            disclaimer_injected=disclaimer_injected,
        )
