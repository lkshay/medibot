"""FAISS-backed vector store for MediBot's disease documents.

For a corpus of ~41 documents, an exact inner-product index (IndexFlatIP)
is the right call: it gives ground-truth rankings and is negligible in
memory. Approximate indexes (HNSW, IVF-PQ) would only pay off at >100k
vectors.

Persisted layout on disk:
    {path}/index.faiss   — raw FAISS binary
    {path}/meta.json     — parallel list of per-vector metadata + provider name
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from .embeddings import EmbeddingProvider


@dataclass
class SearchHit:
    """One retrieval result."""

    disease: str
    document: str
    score: float  # cosine similarity in [-1, 1]; for unit vectors == inner product


class FaissStore:
    """Thin wrapper over a FAISS flat-IP index paired with per-vector metadata."""

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.documents: list[str] = []
        self.metadata: list[dict] = []

    # ------------------------------------------------------------------ build
    def build(self, documents: Sequence[str], metadata: Sequence[dict]) -> None:
        if len(documents) != len(metadata):
            raise ValueError("documents and metadata must be the same length")

        vectors = self.embedder.embed(list(documents))
        if vectors.shape[1] != self.embedder.dim:
            raise ValueError(
                f"embedder reported dim={self.embedder.dim} but returned {vectors.shape[1]}"
            )

        index = faiss.IndexFlatIP(self.embedder.dim)
        index.add(vectors)

        self.index = index
        self.documents = list(documents)
        self.metadata = [dict(m) for m in metadata]

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        q = self.embedder.embed([query])
        scores, ids = self.index.search(q, k)
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0:
                continue
            hits.append(SearchHit(
                disease=self.metadata[idx]["disease"],
                document=self.documents[idx],
                score=float(score),
            ))
        return hits

    # ------------------------------------------------------------------ persist
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Cannot save an empty store.")
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "meta.json").write_text(json.dumps({
            "provider": self.embedder.name,
            "dim": self.embedder.dim,
            "documents": self.documents,
            "metadata": self.metadata,
        }, indent=2))

    def load(self, path: Path | str) -> None:
        path = Path(path)
        self.index = faiss.read_index(str(path / "index.faiss"))
        payload = json.loads((path / "meta.json").read_text())
        if payload["provider"] != self.embedder.name:
            raise ValueError(
                f"Index was built with provider={payload['provider']} but current "
                f"embedder is {self.embedder.name}. Rebuild to avoid dim/semantic mismatch."
            )
        self.documents = payload["documents"]
        self.metadata = payload["metadata"]
