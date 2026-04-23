"""Prompt-injection detection (rule-based baseline).

A production system would layer this with an LLM classifier (or a model like
Meta Llama Guard). For a capstone demo, a curated pattern list catches the
vast majority of naïve injection attempts and teaches the concept clearly.

Patterns covered:
    - Classic 'ignore previous instructions'
    - Role hijacking ('You are now ...', 'Assistant:')
    - System-prompt extraction attempts
    - Jailbreak keywords
"""

from __future__ import annotations

import re

from .schemas import InjectionFinding


# (pattern, severity) — we match case-insensitively.
_PATTERNS: list[tuple[str, str, str]] = [
    (r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|rules|prompts|system)", "high", "override_attempt"),
    (r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)", "high", "override_attempt"),
    (r"forget\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|rules)", "high", "override_attempt"),
    # "you are now" is highly unusual in medical queries; escalate to high.
    (r"you\s+are\s+now\b", "high", "role_hijack"),
    (r"pretend\s+(?:you\s+are|to\s+be)\b", "medium", "role_hijack"),
    (r"act\s+as\s+(?:if\s+you\s+are|a|an)\b", "medium", "role_hijack"),
    (r"(?:^|\n)\s*(?:system|assistant)\s*:", "high", "role_injection"),
    (r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions|rules)", "medium", "prompt_extraction"),
    (r"what\s+(?:are|were)\s+your\s+(?:system\s+)?(?:instructions|rules|prompts)", "medium", "prompt_extraction"),
    (r"repeat\s+(?:the\s+)?(?:above|previous|system)\s+(?:instructions|prompt)", "medium", "prompt_extraction"),
    # Cover "jailbreak" + morphological variants ("jailbroken", "jailbroke").
    (r"\bjail(?:break|breaking|breaker|broke[nd]?)\b", "high", "jailbreak_keyword"),
    (r"\bDAN\s+(?:mode|version|persona)\b", "high", "jailbreak_keyword"),
    (r"(?:new|updated)\s+instructions\s*:", "high", "override_attempt"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), sev, label) for p, sev, label in _PATTERNS]


def detect_injection(text: str) -> list[InjectionFinding]:
    """Return a list of findings. Empty list means 'clean'."""
    findings: list[InjectionFinding] = []
    for pattern, severity, label in _COMPILED:
        m = pattern.search(text)
        if m:
            findings.append(
                InjectionFinding(
                    pattern=label,
                    matched_text=m.group(0),
                    severity=severity,
                )
            )
    return findings
