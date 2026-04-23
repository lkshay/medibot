"""MediBotApp — the top-level entry point wiring together:

    MedicalData + FAISS index + 4 agents + LLM + ReAct orchestrator
    + session/cross-session memory + input & output guardrails

The Gradio UI (M9), the MCP server (bonus), and any integration test
should all go through this one class. Single seam for all swaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .agents import (
    DescriptionAgent,
    DescriptionInput,
    DiagnosisAgent,
    DiagnosisInput,
    PrecautionAgent,
    PrecautionInput,
    SeverityAgent,
    SeverityInput,
)
from .data import MedicalData, load_medical_data
from .guardrails import InputGuard, InputGuardResult, OutputGuard, OutputGuardResult
from .llm import get_llm
from .llm.base import LLMProvider
from .memory import ConversationManager
from .observability import get_tracer
from .orchestrator import ReactOrchestrator, ReactRun, ReactStep, Tool
from .rag.embeddings import LocalEmbedding
from .rag.vector_store import FaissStore


@dataclass
class AskResult:
    """Output of MediBotApp.ask(). Exposes the raw orchestrator run plus
    guardrail findings for observability and debugging."""

    query_original: str
    query_sanitized: str
    input_guard: InputGuardResult
    run: ReactRun | None
    output_guard: OutputGuardResult | None
    blocked: bool
    trace_id: str | None = None  # Langfuse trace id (if observability is enabled)

    @property
    def final_answer(self) -> str:
        if self.blocked:
            return self.input_guard.rejection_reason or "Your request could not be processed."
        if self.output_guard is not None:
            return self.output_guard.final_answer
        if self.run is not None and self.run.final_answer:
            return self.run.final_answer
        return "MediBot did not produce an answer."

    def transcript(self) -> str:
        if self.blocked:
            return (
                f"User: {self.query_original}\n"
                f"[BLOCKED by input guard: {self.input_guard.rejection_reason}]"
            )
        return self.run.transcript() if self.run else ""


@dataclass
class MediBotApp:
    data: MedicalData
    store: FaissStore
    llm: LLMProvider
    orchestrator: ReactOrchestrator
    input_guard: InputGuard
    output_guard: OutputGuard
    sessions: dict[str, ConversationManager] = field(default_factory=dict)

    # ---------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        data_dir: str | Path = "data/raw",
        embedding_path: str | Path = "data/embeddings/faiss_bge_small",
        llm_provider: str | None = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        max_iters: int = 6,
        temperature: float = 0.1,
        enable_phi_redaction: bool = True,
    ) -> "MediBotApp":
        data = load_medical_data(data_dir)

        embedder = LocalEmbedding(embedding_model)
        store = FaissStore(embedder)
        store.load(embedding_path)

        agents = {
            "diagnosis": DiagnosisAgent(store, data),
            "severity": SeverityAgent(data),
            "description": DescriptionAgent(data),
            "precaution": PrecautionAgent(data),
        }
        input_models = {
            "diagnosis": DiagnosisInput,
            "severity": SeverityInput,
            "description": DescriptionInput,
            "precaution": PrecautionInput,
        }
        tools = [
            Tool(
                name=name,
                description=agent.description,
                input_model=input_models[name],
                call=agent,
            )
            for name, agent in agents.items()
        ]

        llm = get_llm(llm_provider)
        orchestrator = ReactOrchestrator(
            llm=llm, tools=tools, max_iters=max_iters, temperature=temperature
        )
        return cls(
            data=data,
            store=store,
            llm=llm,
            orchestrator=orchestrator,
            input_guard=InputGuard(redact_phi=enable_phi_redaction),
            output_guard=OutputGuard(),
        )

    # --------------------------------------------------------------- session
    def session(self, user_id: str) -> ConversationManager:
        if user_id not in self.sessions:
            self.sessions[user_id] = ConversationManager(user_id=user_id)
        return self.sessions[user_id]

    def reset_session(self, user_id: str) -> None:
        if user_id in self.sessions:
            self.sessions[user_id].history = []

    # --------------------------------------------------------------- ask
    def ask(self, user_id: str, query: str) -> AskResult:
        """Full pipeline for one user turn: input guard → memory-aware ReAct → output guard.

        The whole turn is wrapped in a Langfuse trace of type `agent`, with nested
        spans for each guard pass and every ReAct step. user_id is propagated as
        a trace attribute so Langfuse dashboards can filter by user.
        """
        tracer = get_tracer()

        with tracer.trace_attributes(
            user_id=user_id,
            session_id=user_id,  # one session per user for this demo; separate if you add multi-session
            tags=["medibot", self.llm.name],
        ), tracer.span(
            name="medibot.ask",
            as_type="agent",
            input={"query": query, "user_id": user_id},
            metadata={"llm": self.llm.name},
        ) as root:
            trace_id = tracer.current_trace_id()

            # 1. Input guard
            with tracer.span(
                name="input_guard",
                as_type="guardrail",
                input={"text_length": len(query)},
            ) as gs:
                ig = self.input_guard.process(query)
                gs.update(
                    output={
                        "allowed": ig.allowed,
                        "phi_entities": [f.entity_type for f in ig.phi_findings],
                        "injection_patterns": [f.pattern for f in ig.injection_findings],
                    },
                    level="WARNING" if not ig.allowed else "DEFAULT",
                )

            if not ig.allowed:
                root.update(
                    output={"blocked": True, "reason": ig.rejection_reason},
                    level="WARNING",
                )
                return AskResult(
                    query_original=query,
                    query_sanitized=ig.sanitized_text,
                    input_guard=ig,
                    run=None,
                    output_guard=None,
                    blocked=True,
                    trace_id=trace_id,
                )

            # 2. Build conversation context
            conv = self.session(user_id)
            history: list[dict[str, str]] = []
            preamble = conv.build_context_preamble(ig.sanitized_text)
            if preamble:
                history.append({"role": "system", "content": preamble})
            history.extend(conv.history_as_dicts(include_last_n=8))

            # 3. Orchestrator (its own instrumentation adds nested spans)
            run = self.orchestrator.run(ig.sanitized_text, history=history)

            # 4. Output guard
            raw_answer = (
                run.final_answer
                or "I wasn't able to produce a confident answer. Please rephrase or add more symptom details."
            )
            with tracer.span(name="output_guard", as_type="guardrail") as gs:
                og = self.output_guard.process(raw_answer, run)
                gs.update(
                    output={
                        "emergency_injected": og.emergency_injected,
                        "disclaimer_injected": og.disclaimer_injected,
                    }
                )
            run.final_answer = og.final_answer

            # 5. Persist sanitized turn to memory
            conv.add_turn("user", ig.sanitized_text)
            conv.add_turn("assistant", og.final_answer)

            root.update(
                output={
                    "answer": og.final_answer[:500],
                    "iterations": run.iterations,
                    "terminated": run.terminated_reason,
                }
            )

            return AskResult(
                query_original=query,
                query_sanitized=ig.sanitized_text,
                input_guard=ig,
                run=run,
                output_guard=og,
                blocked=False,
                trace_id=trace_id,
            )
