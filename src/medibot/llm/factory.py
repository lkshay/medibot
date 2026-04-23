"""Factory for picking an LLM provider from environment settings.

Env vars (see .env.example):
    LLM_PROVIDER         'gemini' | 'llamacpp' (alias: 'ollama' | 'local') | 'hf'
    GEMINI_MODEL         default 'gemini-2.5-flash'
    OLLAMA_MODEL         default 'gemma4:31b'
    OLLAMA_HOST          default 'http://localhost:11434'
    HF_MODEL             default 'mistralai/Mistral-7B-Instruct-v0.3'
    HF_PROVIDER          optional HF inference router (e.g. 'together', 'fireworks-ai')
    HUGGINGFACEHUB_API_TOKEN / HF_TOKEN
"""

from __future__ import annotations

import os

from .base import LLMProvider
from .gemini import GeminiProvider
from .ollama_client import OllamaProvider


def get_llm(provider: str | None = None) -> LLMProvider:
    """Return an LLMProvider chosen by env var (or explicit arg)."""
    name = (provider or os.environ.get("LLM_PROVIDER") or "gemini").lower().strip()

    if name in {"gemini", "google"}:
        return GeminiProvider(model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    if name in {"llamacpp", "llama.cpp", "ollama", "local"}:
        return OllamaProvider(
            model=os.environ.get("OLLAMA_MODEL", "gemma4:31b"),
            host=os.environ.get("OLLAMA_HOST"),
        )
    if name in {"hf", "huggingface"}:
        from .hf_inference import HFInferenceProvider

        return HFInferenceProvider(
            model=os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
            provider=os.environ.get("HF_PROVIDER"),
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {name!r}")
