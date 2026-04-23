"""Embedding providers behind a single interface.

Two backends:
    LocalEmbedding  — sentence-transformers (default BAAI/bge-small-en-v1.5)
    GeminiEmbedding — Google Gemini Embedding via google-genai SDK

Both return (N, D) float32 arrays with L2-normalized rows, so a cosine-
similarity search is just an inner-product search on the index side.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract embedding backend. All backends emit unit-normalized vectors."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for logging / index metadata."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns shape (len(texts), dim), float32, L2-normalized."""


class LocalEmbedding(EmbeddingProvider):
    """Runs locally via sentence-transformers. No network, no API key."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        # Lazy import so importing this module doesn't pull torch if unused.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"local:{self._name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,  # unit-length -> inner-prod == cosine sim
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype="float32")


class GeminiEmbedding(EmbeddingProvider):
    """Google Gemini Embedding via the google-genai SDK."""

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        api_key: str | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        from google import genai

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY missing. Set it in .env or pass api_key=...")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._task_type = task_type
        # gemini-embedding-001 defaults to 3072-dim; it supports output_dimensionality
        # override. We keep the default unless the caller wants a different size.
        self._dim = 3072

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"gemini:{self._model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
            config={"task_type": self._task_type},
        )
        vecs = np.asarray([e.values for e in resp.embeddings], dtype="float32")
        # Gemini returns unnormalized; normalize so downstream index is cosine-friendly.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
