"""Langfuse-backed tracing facade with a silent no-op fallback.

Design goals:
    - Zero code change in callers when Langfuse is not configured
      (calls become context-managed no-ops, tests stay deterministic)
    - Use Langfuse's semantic span types (`agent`, `tool`, `retriever`,
      `guardrail`, `generation`) so the trace is self-documenting
    - Pass trace_id back to the UI so feedback (thumbs up/down) can score
      the exact turn

Enable by setting in .env:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com    # or a self-hosted URL
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Iterator

log = logging.getLogger(__name__)


class _NoopHandle:
    """Stand-in span that drops all updates on the floor."""

    id: str | None = None

    def update(self, **_: Any) -> None: ...
    def end(self, **_: Any) -> None: ...


class _Handle:
    """Thin wrapper that forwards updates to the underlying Langfuse object."""

    def __init__(self, obs: Any) -> None:
        self._obs = obs

    @property
    def id(self) -> str | None:
        return getattr(self._obs, "id", None)

    def update(self, **kwargs: Any) -> None:
        try:
            self._obs.update(**kwargs)
        except Exception as e:  # noqa: BLE001 — never let telemetry crash the app
            log.debug("span.update failed: %s", e)


class Tracer:
    """Singleton-friendly tracing facade."""

    def __init__(self) -> None:
        self._client = None
        self._enabled = False
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
        sk = os.environ.get("LANGFUSE_SECRET_KEY")
        if not (pk and sk):
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=pk,
                secret_key=sk,
                host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            if self._client.auth_check():
                self._enabled = True
                log.info("Langfuse tracing enabled (host=%s)",
                         os.environ.get("LANGFUSE_HOST", "cloud.langfuse.com"))
            else:
                log.warning("Langfuse auth_check failed — keys invalid? Disabling tracing.")
                self._client = None
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to initialize Langfuse: %s", e)
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------ spans
    @contextlib.contextmanager
    def trace_attributes(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ):
        """Propagate user_id / session_id / tags to the current (and all child) traces.

        Wrap your root span in this context. Langfuse attributes propagate
        through OpenTelemetry context, so no-ops on the outer nested spans.
        """
        if not self._enabled:
            yield
            return
        try:
            with self._client.propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                tags=tags or None,
            ):
                yield
        except Exception as e:  # noqa: BLE001
            log.debug("propagate_attributes failed: %s", e)
            yield

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        as_type: str = "span",
        input: Any = None,
        metadata: dict | None = None,
    ) -> Iterator[_Handle | _NoopHandle]:
        """Start a nested observation of the given semantic type.

        as_type ∈ {"span", "agent", "tool", "chain", "retriever", "evaluator",
                   "guardrail", "generation", "embedding"}
        """
        if not self._enabled:
            yield _NoopHandle()
            return
        try:
            ctx = self._client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=input,
                metadata=metadata,
            )
        except Exception as e:  # noqa: BLE001
            log.debug("tracer.span start failed: %s", e)
            yield _NoopHandle()
            return
        with ctx as obs:
            yield _Handle(obs)

    @contextlib.contextmanager
    def generation(
        self,
        name: str,
        model: str,
        input: Any = None,
        metadata: dict | None = None,
    ) -> Iterator[_Handle | _NoopHandle]:
        """Start an LLM-call generation span (extra fields for model + tokens)."""
        if not self._enabled:
            yield _NoopHandle()
            return
        try:
            ctx = self._client.start_as_current_observation(
                name=name,
                as_type="generation",
                model=model,
                input=input,
                metadata=metadata,
            )
        except Exception as e:  # noqa: BLE001
            log.debug("tracer.generation start failed: %s", e)
            yield _NoopHandle()
            return
        with ctx as obs:
            yield _Handle(obs)

    # ---------------------------------------------------------------- helpers
    def current_trace_id(self) -> str | None:
        if not self._enabled:
            return None
        try:
            return self._client.get_current_trace_id()
        except Exception:  # noqa: BLE001
            return None

    def score_trace(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str = "",
    ) -> None:
        """Attach a named score to a trace (used for user feedback)."""
        if not self._enabled or not trace_id:
            return
        try:
            self._client.create_score(
                name=name,
                value=value,
                trace_id=trace_id,
                comment=comment or None,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Langfuse score_trace failed: %s", e)

    def flush(self) -> None:
        if self._enabled:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------- singleton
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Return the process-wide Tracer (initializes lazily on first call)."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
