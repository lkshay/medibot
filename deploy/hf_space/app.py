"""Hugging Face Spaces entry point.

Launches the MediBot Gradio UI with the HF Inference API as the LLM
backend. The `medibot` package itself is installed from GitHub via
requirements.txt, so this file stays thin.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    # Default to HF Inference API when running in a Space.
    os.environ.setdefault("LLM_PROVIDER", "hf")

    # Build the FAISS index if we haven't yet (persists across restarts in
    # the Space's ephemeral filesystem, so each cold start rebuilds).
    idx_dir = Path("data/embeddings/faiss_bge_small")
    if not idx_dir.exists():
        print("Building FAISS index (one-time, ~30s) ...", flush=True)
        # Inline the minimal build so we don't require the scripts/ dir.
        from medibot.data import disease_to_document, load_medical_data
        from medibot.rag.embeddings import LocalEmbedding
        from medibot.rag.vector_store import FaissStore

        data = load_medical_data("data/raw")
        store = FaissStore(LocalEmbedding("BAAI/bge-small-en-v1.5"))
        store.build(
            [disease_to_document(d, data) for d in data.diseases],
            [{"disease": d} for d in data.diseases],
        )
        store.save(idx_dir)
        print("Index built.", flush=True)

    # HF Spaces expects the Gradio app on 0.0.0.0:7860
    from medibot.ui.gradio_app import launch

    launch(
        llm_provider="hf",
        server_port=int(os.environ.get("PORT", 7860)),
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
