#!/usr/bin/env python
"""Interactive CLI for chatting with MediBot.

Usage:
    python scripts/chat.py                          # local Ollama (gemma4:latest)
    python scripts/chat.py --llm ollama --model gemma4:31b
    python scripts/chat.py --llm gemini             # requires GOOGLE_API_KEY
    python scripts/chat.py --user alice             # set a user id (for memory)
    python scripts/chat.py --trace                  # show the ReAct trace each turn
    python scripts/chat.py "what are symptoms of diabetes"   # one-shot mode

In the REPL:
    quit / exit / q    leave
    /reset             clear session history (cross-session memory kept)
    /trace             toggle trace view
"""

from __future__ import annotations

import argparse
import os
import sys


def _args():
    p = argparse.ArgumentParser(description="Chat with MediBot from the terminal.")
    p.add_argument("query", nargs="*", help="One-shot question; if omitted, enters REPL.")
    p.add_argument("--llm", default="ollama", choices=["ollama", "gemini", "hf"])
    p.add_argument("--model", default=None, help="Override model (OLLAMA_MODEL / GEMINI_MODEL / HF_MODEL)")
    p.add_argument("--user", default="you", help="User id for memory isolation")
    p.add_argument("--trace", action="store_true", help="Print the ReAct trace each turn")
    return p.parse_args()


def _meta_line(result) -> str:
    bits = []
    if result.input_guard.phi_findings:
        bits.append(f"PHI redacted: {[f.entity_type for f in result.input_guard.phi_findings]}")
    if result.input_guard.injection_findings:
        bits.append(f"injection signals: {[f.pattern for f in result.input_guard.injection_findings]}")
    if result.output_guard:
        if result.output_guard.emergency_injected:
            bits.append("emergency banner injected")
        if result.output_guard.disclaimer_injected:
            bits.append("disclaimer injected")
    if result.run:
        bits.append(f"{result.run.iterations} step(s)")
    return "; ".join(bits)


def _print_result(result, show_trace: bool) -> None:
    print()
    if result.blocked:
        print(f"[BLOCKED] {result.input_guard.rejection_reason}")
        print()
        return
    if show_trace:
        print("--- trace ---")
        print(result.transcript())
        print("-" * 60)
    print("medibot>")
    print(result.final_answer)
    meta = _meta_line(result)
    if meta:
        print(f"\n[{meta}]")
    print()


def main() -> int:
    # Load .env if present so API keys / Langfuse creds pick up automatically.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    args = _args()
    if args.model:
        env_var = {"ollama": "OLLAMA_MODEL", "gemini": "GEMINI_MODEL", "hf": "HF_MODEL"}[args.llm]
        os.environ[env_var] = args.model

    print(f"Loading MediBot (llm={args.llm})...", flush=True)
    from medibot.app import MediBotApp  # import after env vars set

    app = MediBotApp.build(llm_provider=args.llm)
    print(f"Ready. LLM: {app.llm.name}. user={args.user}.\n")

    # One-shot mode
    if args.query:
        q = " ".join(args.query)
        result = app.ask(args.user, q)
        _print_result(result, show_trace=args.trace)
        return 0

    # Interactive REPL
    show_trace = args.trace
    print("Type 'quit' to exit, '/reset' to clear history, '/trace' to toggle trace.\n")
    while True:
        try:
            query = input(f"you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        if query == "/reset":
            app.reset_session(args.user)
            print("[session reset]\n")
            continue
        if query == "/trace":
            show_trace = not show_trace
            print(f"[trace: {'on' if show_trace else 'off'}]\n")
            continue

        try:
            result = app.ask(args.user, query)
        except Exception as e:  # noqa: BLE001
            print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
            continue

        _print_result(result, show_trace=show_trace)

    return 0


if __name__ == "__main__":
    sys.exit(main())
