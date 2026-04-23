"""LLM provider abstraction.

Every backend exposes the same two operations:

    generate(prompt, ...)    — simple completion, returns a str
    chat(messages, ...)      — multi-turn; messages is a list of Message

The abstraction hides SDK differences between cloud (Gemini) and local
(Ollama / llama.cpp) backends. Downstream agents depend on `LLMProvider`
alone, so switching LLMs means one env var, not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """One chat turn. Intentionally minimal — we parse ReAct as text."""

    role: Role
    content: str


@dataclass
class GenerationConfig:
    """Common generation knobs. Backends ignore what they don't support."""

    temperature: float = 0.2
    max_output_tokens: int = 2048
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)


class LLMProvider(ABC):
    """Every backend implements these. No tool/function-calling in the
    interface — the ReAct orchestrator (Milestone 4) handles tool dispatch
    by parsing plain text, which keeps the abstraction backend-agnostic."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier: 'gemini:gemini-2.5-flash', 'ollama:gemma4:31b'."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        config: GenerationConfig | None = None,
    ) -> str:
        """Single-turn completion."""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        """Multi-turn. System message (if any) must be the first message."""
