"""Promptfoo Python-provider adapter for MediBotApp.

Promptfoo calls `call_api(prompt, options, context)` per test case. We build
the app once at import time (cached), then run `ask()` per request.
"""

from __future__ import annotations

import os
from functools import lru_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@lru_cache(maxsize=4)
def _build_app(llm_provider: str):
    from medibot.app import MediBotApp
    return MediBotApp.build(llm_provider=llm_provider)


def call_api(prompt: str, options: dict | None = None, context: dict | None = None):
    """Entry point called by Promptfoo for each test case."""
    options = options or {}
    config = options.get("config") or {}
    llm = config.get("llm") or os.environ.get("LLM_PROVIDER") or "gemini"

    app = _build_app(llm)

    # Prefer the `query` var if set, else use the raw prompt
    query = (context or {}).get("vars", {}).get("query", prompt)
    result = app.ask(user_id="promptfoo", query=query)

    return {
        "output": result.final_answer,
        "cost": 0,
        "tokenUsage": {
            "prompt": 0,
            "completion": 0,
        },
        "metadata": {
            "blocked": result.blocked,
            "iterations": result.run.iterations if result.run else 0,
            "tools_called": [
                s.action for s in (result.run.steps if result.run else []) if s.action
            ],
        },
    }


def medibot(prompt: str, options: dict | None = None, context: dict | None = None):
    """Alias used in promptfooconfig.yaml (Promptfoo function-style providers)."""
    return call_api(prompt, options, context)
