#!/usr/bin/env python
"""Build (or rebuild) the FAISS vector index from the raw CSV data.

Run this once after cloning:
    python scripts/build_index.py

Artifacts are written to data/embeddings/faiss_bge_small/ (gitignored).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from medibot.data import disease_to_document, load_medical_data
    from medibot.rag.embeddings import LocalEmbedding
    from medibot.rag.vector_store import FaissStore

    out_dir = Path("data/embeddings/faiss_bge_small")
    print(f"Loading medical data ...")
    data = load_medical_data("data/raw")
    print(f"  {len(data.diseases)} diseases, {len(data.symptom_severity)} symptoms")

    print(f"Loading embedder (BAAI/bge-small-en-v1.5) ...")
    embedder = LocalEmbedding("BAAI/bge-small-en-v1.5")

    print(f"Building FAISS IndexFlatIP (dim={embedder.dim}) ...")
    store = FaissStore(embedder)
    documents = [disease_to_document(d, data) for d in data.diseases]
    metadata = [{"disease": d} for d in data.diseases]
    store.build(documents, metadata)

    print(f"Saving to {out_dir}/")
    store.save(out_dir)

    # Sanity check: recall@3 on synthetic queries
    K = 3
    hits = 0
    for d in data.diseases:
        syms = list(data.disease_symptoms[d])[:4]
        q = "I have " + ", ".join(s.replace("_", " ") for s in syms)
        results = store.search(q, k=K)
        if any(r.disease == d for r in results):
            hits += 1
    rate = hits / len(data.diseases)
    print(f"Sanity check: recall@{K} = {hits}/{len(data.diseases)} = {rate:.1%}")
    if rate < 0.95:
        print("WARNING: retrieval quality regressed below 95%!", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
