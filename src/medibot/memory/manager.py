"""ConversationManager — unifies session + cross-session memory.

Session memory: an in-process buffer of the current conversation's turns.
Cross-session memory: Mem0 (hosted or self-hosted), if configured.

Why both:
- Session memory gives the agent short-term context (the user said
  'itching and rash' three turns ago; the current 'what precautions?'
  refers to that).
- Cross-session memory gives the agent long-term personalization (the
  user mentioned last week that they have diabetes; a current symptom
  query should take that into account).

If MEM0_API_KEY is not set (or mem0ai is not installed), the manager
gracefully degrades to session-only. This keeps local dev frictionless.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """Per-user conversation state + optional Mem0-backed long-term memory."""

    def __init__(
        self,
        user_id: str,
        enable_mem0: bool = True,
        max_session_turns: int = 20,
    ) -> None:
        self.user_id = user_id
        self.history: list[ConversationTurn] = []
        self.max_session_turns = max_session_turns
        self._mem0_client = None

        if enable_mem0 and os.environ.get("MEM0_API_KEY"):
            try:
                from mem0 import MemoryClient  # type: ignore

                self._mem0_client = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
                log.info("Mem0 cross-session memory enabled for user_id=%s", user_id)
            except ImportError:
                log.warning("MEM0_API_KEY set but mem0ai package is not installed.")
            except Exception as e:  # noqa: BLE001 — keep the app alive without Mem0
                log.warning("Failed to initialize Mem0: %s", e)

    # ------------------------------------------------------------------ session
    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a turn to session history and (optionally) push to Mem0."""
        self.history.append(ConversationTurn(role=role, content=content, metadata=metadata or {}))

        # Keep session history bounded
        if len(self.history) > self.max_session_turns:
            self.history = self.history[-self.max_session_turns:]

        # Persist user turns to Mem0 (Mem0 extracts facts internally)
        if self._mem0_client and role == "user":
            try:
                self._mem0_client.add(
                    messages=[{"role": role, "content": content}],
                    user_id=self.user_id,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Mem0 add failed: %s", e)

    def history_as_dicts(self, include_last_n: int | None = None) -> list[dict[str, str]]:
        """Return session history in the {role, content} dict form our orchestrator expects."""
        turns = self.history[-include_last_n:] if include_last_n else self.history
        return [{"role": t.role, "content": t.content} for t in turns]

    # ---------------------------------------------------------- cross-session
    def retrieve_memories(self, query: str, k: int = 3) -> list[str]:
        """Semantic retrieval of prior facts about this user from Mem0."""
        if not self._mem0_client:
            return []
        try:
            hits = self._mem0_client.search(
                query=query,
                user_id=self.user_id,
                limit=k,
            )
            return [h.get("memory") or h.get("text", "") for h in hits if h]
        except Exception as e:  # noqa: BLE001
            log.warning("Mem0 search failed: %s", e)
            return []

    def build_context_preamble(self, query: str) -> str:
        """Optional preamble injected ahead of the current question, summarizing
        cross-session memories relevant to the query. Returns '' if no memories."""
        memories = self.retrieve_memories(query)
        if not memories:
            return ""
        bullets = "\n".join(f"- {m}" for m in memories)
        return f"Relevant facts from previous conversations with this user:\n{bullets}"
