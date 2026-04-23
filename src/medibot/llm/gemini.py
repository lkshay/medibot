"""Google Gemini provider using the google-genai SDK.

The unified SDK (`from google import genai`) replaces the older
`google-generativeai` package in 2025+. API key comes from the
`GOOGLE_API_KEY` env var (see `.env.example`).
"""

from __future__ import annotations

import os

from .base import GenerationConfig, LLMProvider, Message


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        from google import genai

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set; cannot initialize GeminiProvider.")
        self._client = genai.Client(api_key=key)
        self._model = model

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    def _config(self, system: str, config: GenerationConfig | None):
        from google.genai import types

        cfg = config or GenerationConfig()
        return types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            top_p=cfg.top_p,
            stop_sequences=cfg.stop or None,
        )

    def generate(
        self,
        prompt: str,
        system: str = "",
        config: GenerationConfig | None = None,
    ) -> str:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._config(system, config),
        )
        return (resp.text or "").strip()

    def chat(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        from google.genai import types

        # Peel off the system message, if any, and fold it into the config.
        system_text = ""
        turns = list(messages)
        if turns and turns[0].role == "system":
            system_text = turns[0].content
            turns = turns[1:]

        # Gemini uses role="user" and role="model"; map ours onto theirs.
        role_map = {"user": "user", "assistant": "model"}
        contents = [
            types.Content(role=role_map[m.role], parts=[types.Part(text=m.content)])
            for m in turns
        ]
        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=self._config(system_text, config),
        )
        return (resp.text or "").strip()
