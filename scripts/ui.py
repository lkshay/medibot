#!/usr/bin/env python
"""Launcher for the MediBot Gradio UI.

    python scripts/ui.py                          # local Ollama (gemma4:latest)
    python scripts/ui.py --llm ollama --model gemma4:31b
    python scripts/ui.py --llm gemini             # requires GOOGLE_API_KEY
    python scripts/ui.py --share                  # create a public tunneled URL
    python scripts/ui.py --port 7861
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    # Load .env if present so users can configure API keys without exporting.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    p = argparse.ArgumentParser(description="Launch the MediBot Gradio UI.")
    p.add_argument("--llm", default="ollama", choices=["ollama", "gemini", "hf"])
    p.add_argument("--model", default=None, help="Override model (OLLAMA_MODEL / GEMINI_MODEL / HF_MODEL).")
    p.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="Server bind address.")
    args = p.parse_args()

    if args.model:
        env_var = {
            "ollama": "OLLAMA_MODEL",
            "gemini": "GEMINI_MODEL",
            "hf": "HF_MODEL",
        }[args.llm]
        os.environ[env_var] = args.model

    from medibot.ui.gradio_app import launch

    launch(
        llm_provider=args.llm,
        share=args.share,
        server_port=args.port,
        server_name=args.host,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
