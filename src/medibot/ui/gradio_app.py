"""Gradio web UI for MediBot.

A single-page chat interface with:
    - Multi-turn chat (session memory via MediBotApp)
    - Optional collapsible reasoning-trace panel
    - Guardrail-findings panel (PHI redactions, injection flags)
    - Thumbs-up/down feedback hook (stubbed now, wired to Langfuse in M7)
    - Per-browser anonymous user id (so memory isolates between tabs)
    - Examples, reset button, model-info header

Uses gr.Blocks (not ChatInterface) so we can expose the trace and guard
panels alongside the chat.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass

import gradio as gr

from ..app import AskResult, MediBotApp
from ..observability import get_tracer


log = logging.getLogger(__name__)


_DESCRIPTION = """# 🩺 MediBot

Describe your symptoms in plain English. MediBot reasons step-by-step using four specialized tools
(diagnosis, severity, description, precaution) and returns grounded answers.

> **⚠️ Informational only — not medical advice.** For emergencies, call your local emergency services.
"""

_EXAMPLES = [
    "I have itching and a red skin rash with patches for 3 days. What could this be?",
    "What precautions should I take for chicken pox?",
    "What are the symptoms of diabetes?",
    "I have excessive thirst, frequent urination, and unexplained weight loss.",
    "I have sudden sharp chest pain and trouble breathing.",
]


def _format_trace(result: AskResult) -> str:
    if result.blocked or result.run is None:
        return ""
    return f"```\n{result.run.transcript()}\n```"


def _format_guards(result: AskResult) -> dict:
    """Flatten guard findings into a JSON-display-friendly dict."""
    ig = result.input_guard
    og = result.output_guard
    return {
        "blocked_by_input_guard": result.blocked,
        "rejection_reason": ig.rejection_reason,
        "phi_redactions": [
            {"type": f.entity_type, "original": f.original, "score": round(f.score, 3)}
            for f in ig.phi_findings
        ],
        "injection_signals": [
            {"pattern": f.pattern, "severity": f.severity, "text": f.matched_text}
            for f in ig.injection_findings
        ],
        "emergency_banner_injected": og.emergency_injected if og else None,
        "disclaimer_injected": og.disclaimer_injected if og else None,
        "react_iterations": result.run.iterations if result.run else 0,
        "terminated_reason": result.run.terminated_reason if result.run else None,
    }


def build_ui(app: MediBotApp) -> gr.Blocks:
    with gr.Blocks(
        title="MediBot — AI Symptom Checker",
        fill_height=True,
    ) as demo:
        # --- state ---------------------------------------------------------
        session_id = gr.State(value="")  # set on load
        last_result = gr.State(value=None)  # last AskResult for feedback handler

        # --- header --------------------------------------------------------
        gr.Markdown(_DESCRIPTION)
        gr.Markdown(
            f"_LLM backend:_ **{app.llm.name}** · _Corpus:_ "
            f"**{len(app.data.diseases)} diseases / {len(app.data.symptom_severity)} symptoms**"
        )

        with gr.Row():
            # --- main column (chat) ---------------------------------------
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="MediBot",
                    height=520,
                    placeholder="Your conversation with MediBot will appear here.",
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Describe your symptoms or ask a medical question...",
                        show_label=False,
                        scale=8,
                        autofocus=True,
                        max_lines=4,
                    )
                    submit_btn = gr.Button("Send ➤", variant="primary", scale=1)

                with gr.Row():
                    reset_btn = gr.Button("🧹 Reset conversation", size="sm")
                    show_trace = gr.Checkbox(
                        label="Show reasoning trace",
                        value=False,
                    )

                gr.Examples(examples=_EXAMPLES, inputs=user_input)

            # --- right column (inspection) --------------------------------
            with gr.Column(scale=2):
                with gr.Accordion("🔎 Reasoning trace", open=False) as trace_panel:
                    trace_md = gr.Markdown(
                        value="_Toggle 'Show reasoning trace' to see the step-by-step ReAct loop._",
                    )
                with gr.Accordion("🛡️ Guardrail findings", open=False):
                    guard_json = gr.JSON(
                        value={},
                        label="Latest turn",
                    )
                with gr.Accordion("ℹ️ About", open=False):
                    gr.Markdown(
                        "MediBot is a multi-agent medical symptom checker built with LangChain-style "
                        "ReAct (from scratch), FAISS vector retrieval, Pydantic-typed tools, Presidio "
                        "PHI redaction, and a dual LLM backend (Gemini cloud / local Gemma4 via "
                        "llama.cpp/Ollama).\n\n"
                        "This is a capstone demo. Not a medical device."
                    )

        # --- event handlers ------------------------------------------------
        def _on_load() -> str:
            # Per-browser session id; prefixed so it's easy to find in logs.
            return f"ui-{uuid.uuid4().hex[:8]}"

        def chat_fn(
            message: str,
            history: list,
            user_id: str,
            show_trace_flag: bool,
        ):
            if not message or not message.strip():
                return history, "", gr.skip(), gr.skip(), gr.skip()

            t0 = time.time()
            try:
                result = app.ask(user_id, message)
            except Exception as e:  # noqa: BLE001
                log.exception("app.ask failed")
                err = f"**⚠️ Internal error:** {type(e).__name__}: {e}"
                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": err},
                ]
                return history, "", "", {"error": str(e)}, None

            elapsed = time.time() - t0
            log.info("turn user=%s elapsed=%.2fs steps=%s",
                     user_id,
                     elapsed,
                     result.run.iterations if result.run else "blocked")

            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result.final_answer},
            ]

            trace = _format_trace(result) if show_trace_flag else \
                "_Toggle 'Show reasoning trace' to see the step-by-step ReAct loop._"
            guards = _format_guards(result)
            return history, "", trace, guards, result

        submit_btn.click(
            chat_fn,
            inputs=[user_input, chatbot, session_id, show_trace],
            outputs=[chatbot, user_input, trace_md, guard_json, last_result],
        )
        user_input.submit(
            chat_fn,
            inputs=[user_input, chatbot, session_id, show_trace],
            outputs=[chatbot, user_input, trace_md, guard_json, last_result],
        )

        def reset_fn(user_id: str):
            app.reset_session(user_id)
            return [], "_Session reset._", {}, None

        reset_btn.click(
            reset_fn,
            inputs=[session_id],
            outputs=[chatbot, trace_md, guard_json, last_result],
        )

        # Thumbs up/down → Langfuse score (no-op if Langfuse isn't configured).
        def on_like(evt: gr.LikeData, user_id: str, result: AskResult | None):
            if result is None:
                return
            value = 1.0 if evt.liked else 0.0
            log.info(
                "feedback user=%s value=%s trace_id=%s",
                user_id,
                value,
                (result.trace_id or "-"),
            )
            tracer = get_tracer()
            if result.trace_id:
                tracer.score_trace(
                    trace_id=result.trace_id,
                    name="user_feedback",
                    value=value,
                    comment=f"user={user_id}",
                )

        chatbot.like(on_like, inputs=[session_id, last_result], outputs=None)

        demo.load(_on_load, outputs=session_id)

    return demo


def launch(
    llm_provider: str | None = None,
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1",
) -> None:
    """Build the app + UI and launch the Gradio server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print("Loading MediBot (this takes a moment)...", flush=True)
    app = MediBotApp.build(llm_provider=llm_provider)
    print(f"Ready. Starting Gradio on http://{server_name}:{server_port} ...")
    demo = build_ui(app)
    demo.queue(max_size=16)
    demo.launch(
        share=share,
        server_port=server_port,
        server_name=server_name,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="gray"),
    )
