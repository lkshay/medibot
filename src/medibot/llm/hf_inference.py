"""Hugging Face Inference-API backed LLM provider.

Uses `huggingface_hub.InferenceClient` — a single HF token works against
either HF's free Serverless Inference API (rate-limited) or a user's own
dedicated Inference Endpoint.

Intended use: HF Spaces deployment where spinning up Ollama is too heavy
for the free CPU tier. At home you can still use the Ollama or Gemini
backends via LLM_PROVIDER.
"""

from __future__ import annotations

import os

from .base import GenerationConfig, LLMProvider, Message


class HFInferenceProvider(LLMProvider):
    """HF Inference API via huggingface_hub.InferenceClient."""

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        token: str | None = None,
        timeout: float = 120.0,
        provider: str | None = None,
    ) -> None:
        from huggingface_hub import InferenceClient

        token = (
            token
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HF_TOKEN")
        )
        if not token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) must be set to use HFInferenceProvider."
            )
        self._model = model
        self._client = InferenceClient(
            model=model,
            token=token,
            timeout=timeout,
            provider=provider,  # e.g. "hf-inference", "together", "fireworks-ai"
        )

    @property
    def name(self) -> str:
        return f"hf:{self._model}"

    def generate(
        self,
        prompt: str,
        system: str = "",
        config: GenerationConfig | None = None,
    ) -> str:
        """Use chat_completion under the hood so instruct templates apply correctly."""
        cfg = config or GenerationConfig()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat_completion(
            messages=messages,
            max_tokens=cfg.max_output_tokens,
            temperature=max(0.01, float(cfg.temperature)),  # HF rejects temp==0
            top_p=float(cfg.top_p),
            stop=cfg.stop or None,
        )
        return (resp.choices[0].message.content or "").strip()

    def chat(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        cfg = config or GenerationConfig()
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        resp = self._client.chat_completion(
            messages=chat_messages,
            max_tokens=cfg.max_output_tokens,
            temperature=max(0.01, float(cfg.temperature)),
            top_p=float(cfg.top_p),
            stop=cfg.stop or None,
        )
        return (resp.choices[0].message.content or "").strip()
