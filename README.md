# MediBot

An AI-powered multi-agent medical symptom checker. Built with a from-scratch ReAct orchestrator, FAISS retrieval, Presidio PHI redaction, and a dual LLM backend (Gemini cloud / Gemma4 local via llama.cpp/Ollama). Observability via Langfuse.

## Stack

| Layer | Technology |
|---|---|
| Agent framework | Custom ReAct orchestrator (Python, ~300 LOC) |
| LLM providers | Google Gemini (cloud) + Gemma4 via Ollama/llama.cpp (local) |
| Embeddings | BAAI/bge-small-en-v1.5 via sentence-transformers |
| Vector store | FAISS (IndexFlatIP) |
| Memory | In-session buffer + Mem0 cross-session (optional) |
| Guardrails | Microsoft Presidio (PHI) + rule-based prompt-injection + output policy |
| Observability | Langfuse (traces, spans, token counts, feedback scores) |
| Evaluation | Custom + LLM-as-judge (RAGAS-style faithfulness/relevancy) |
| CI | pytest + GitHub Actions (+ Promptfoo for agent evals) |
| UI | Gradio (Blocks) |

## Quick start

```bash
# 1. Create venv and install
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install sentence-transformers faiss-cpu pandas pyyaml \
  presidio-analyzer presidio-anonymizer httpx pydantic python-dotenv \
  google-genai mem0ai langfuse gradio pytest
python -m spacy download en_core_web_lg

# 2. Configure (optional — defaults work for local-only)
cp .env.example .env
# Edit .env to add GOOGLE_API_KEY (for Gemini), LANGFUSE_PUBLIC_KEY+SECRET (for tracing)

# 3. Build the vector index (one-time, ~30s)
python scripts/build_index.py

# 4. Pull a local LLM (if going local)
ollama pull gemma3:latest   # or gemma3:27b for more capacity

# 5. Run
python scripts/chat.py                  # CLI REPL
python scripts/ui.py                    # Web UI at http://127.0.0.1:7860
python scripts/run_evals.py --no-judge  # Eval suite (13 curated cases)
python -m pytest tests/                 # Unit tests (73 cases)
```

## Project layout

```
medibot/
├── src/medibot/
│   ├── data.py                 # CSV loader + normalization + alias fixups
│   ├── rag/                    # Embeddings + FAISS store
│   ├── agents/                 # 4 tool agents (diagnosis/severity/description/precaution)
│   ├── llm/                    # Provider abstraction (Gemini + Ollama)
│   ├── orchestrator/           # From-scratch ReAct loop + prompts
│   ├── memory/                 # Session + Mem0 cross-session memory
│   ├── guardrails/             # PHI redaction + prompt-injection + output policy
│   ├── observability/          # Langfuse tracing facade
│   ├── evaluation/             # Metrics + LLM-judge + runner
│   ├── ui/                     # Gradio Blocks app
│   └── app.py                  # MediBotApp — top-level API
├── scripts/
│   ├── chat.py                 # CLI REPL
│   ├── ui.py                   # Launch Gradio
│   ├── run_evals.py            # Eval suite runner
│   ├── build_index.py          # Rebuild FAISS index
│   └── promptfoo_provider.py   # Promptfoo bridge
├── notebooks/                  # EDA + RAG build (teaching artifacts)
├── tests/                      # pytest unit tests (73 cases)
├── evals/
│   ├── eval_set.yaml           # Curated golden set (13 cases)
│   └── reports/                # Generated reports (gitignored)
├── data/raw/                   # Source CSVs
├── promptfooconfig.yaml        # Declarative eval config
└── .github/workflows/ci.yml    # pytest + FAISS recall + (optional) Gemini eval
```

## Architecture at a glance

```
┌── user turn ───────────────────────────────────────────────────┐
│  MediBotApp.ask()                                              │
│  ├── InputGuard  (Presidio PHI redaction, injection block)     │
│  ├── ReactOrchestrator.run()                                   │
│  │   loop up to max_iters:                                     │
│  │     ├── LLM.generate(system + scratchpad)                   │
│  │     ├── parse Thought/Action/ActionInput                    │
│  │     └── dispatch tool → observation → append scratchpad     │
│  │   terminates on "Final Answer:" or max_iters                │
│  ├── OutputGuard (disclaimer, emergency banner enforcement)    │
│  └── persist to session + (optional) Mem0                      │
│                                                                │
│  Every step emits a Langfuse span (type-annotated: agent,      │
│  guardrail, chain, span, tool, retriever, generation) with     │
│  input/output, latency, tokens.                                │
└────────────────────────────────────────────────────────────────┘
```

## Status

- ✅ 73/73 unit tests
- ✅ 13/13 agent eval cases (deterministic metrics)
- ✅ FAISS recall@3 = 100% over 41 diseases
- ✅ Langfuse traces flowing end-to-end
- ✅ Gradio UI with trace + guard inspection panels
- 🔜 HF Spaces deployment
- 🔜 Optional LangGraph variant of the orchestrator
- 🔜 Optional MCP server exposing the 4 agents

## Deploy

Local Gradio is fine for demos. For a public-facing deploy:
- **HF Spaces** (free tier): requires Gemini cloud backend (local Gemma4 won't fit in 16 GB RAM, no GPU)
- **Production path** (documented, not built): FastAPI + Docker + GCP Cloud Run with Vertex AI for Gemini
