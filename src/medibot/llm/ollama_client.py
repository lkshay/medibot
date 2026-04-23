"""Ollama-backed local LLM provider (llama.cpp under the hood).

Ollama runs a local HTTP server that wraps the llama.cpp engine. From our
side it looks like any REST API; the engine itself is the same llama.cpp
binary you'd call directly, just with a cleaner interface and automatic
model management.

Start the server once: `ollama serve` (auto-starts on macOS).
"""

from __future__ import annotations

import os

import httpx

from .base import GenerationConfig, LLMProvider, Message


class OllamaProvider(LLMProvider):
    """Talks to Ollama's /api/generate and /api/chat endpoints."""

    def __init__(
        self,
        model: str = "gemma4:31b",
        host: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self.last_usage: dict | None = None  # populated by generate()/chat() for telemetry

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def _options(self, config: GenerationConfig | None) -> dict:
        cfg = config or GenerationConfig()
        opts = {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_output_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.stop:
            opts["stop"] = cfg.stop
        return opts

    def _record_usage(self, body: dict) -> None:
        """Ollama returns token counts and durations we can feed to Langfuse."""
        input_toks = body.get("prompt_eval_count")
        output_toks = body.get("eval_count")
        if input_toks is None and output_toks is None:
            self.last_usage = None
            return
        self.last_usage = {
            "input": int(input_toks or 0),
            "output": int(output_toks or 0),
            "total": int((input_toks or 0) + (output_toks or 0)),
        }

    def generate(
        self,
        prompt: str,
        system: str = "",
        config: GenerationConfig | None = None,
    ) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(config),
        }
        if system:
            payload["system"] = system
        r = self._client.post(f"{self._host}/api/generate", json=payload)
        r.raise_for_status()
        body = r.json()
        self._record_usage(body)
        return body.get("response", "").strip()

    def chat(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": self._options(config),
        }
        r = self._client.post(f"{self._host}/api/chat", json=payload)
        r.raise_for_status()
        body = r.json()
        self._record_usage(body)
        return body.get("message", {}).get("content", "").strip()
