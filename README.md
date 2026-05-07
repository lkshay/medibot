# MediBot — Capstone Project

An AI-powered multi-agent medical symptom checker built as a learning vehicle for **agentic application development**. The system pairs a from-scratch ReAct orchestrator with FAISS retrieval, Presidio PHI guardrails, a dual LLM backend (Gemini cloud / Gemma3 local), and Langfuse observability.

This README is primarily a record of what I learned and where I'd take this next. For setup details, see the **Appendix** at the bottom.

---

## What this project is

MediBot accepts a free-form symptom description, redacts PHI, runs a Reason–Act–Observe loop over four specialist tool-agents (diagnosis, severity, description, precaution), enforces an output policy (disclaimer + emergency banner), and emits per-step Langfuse traces. It ships with 73 unit tests, a 13-case golden eval set, a Promptfoo config, a Gradio UI, and a CI pipeline.

The interesting part isn't the medical output — it's the **plumbing around the LLM** that makes the system honest, observable, and testable.

---

## Key learnings

### 1. Agentic systems are mostly *not* the LLM

The LLM is maybe 20% of the work. The rest is:

- **Tool contracts** — every agent is gated by a Pydantic input/output schema. The orchestrator validates both sides of the boundary, which catches malformed tool calls long before they cost a downstream hallucination.
- **A scratchpad parser** — the ReAct loop lives or dies on cleanly parsing `Thought / Action / Action Input / Observation`. I learned the hard way that whitespace, stop tokens, and stray code fences are 90% of "agent failures."
- **Termination conditions** — `max_iters`, repeated-action detection, and an explicit `Final Answer:` sentinel are not optional. Without them, the agent will happily loop forever or end on a tool call.
- **Provider abstraction** — swapping Gemini ↔ Gemma3 (Ollama) ↔ llama.cpp through one `LLMProvider` interface forced me to think about generation config, stop sequences, and token accounting as first-class concepts, not vendor details.

### 2. Building ReAct from scratch beats using a framework first

I deliberately wrote the orchestrator in ~300 lines before touching LangGraph. Doing this once means I now read framework code instead of treating it as magic — every node, edge, and state transition in LangGraph maps onto something I had to build by hand (router, scratchpad, tool dispatch, halt condition). Recommended for anyone learning this space.

### 3. Guardrails belong at the boundary, not inside the agent

Splitting input/output into separate `InputGuard` / `OutputGuard` modules — Presidio for PHI, rule-based for prompt injection, an output policy for disclaimers — kept the orchestrator code about reasoning, not safety. When a guard tripped during evals, I knew exactly where to look.

### 4. Observability changes how you debug agents

Before Langfuse: I read raw logs and guessed why the agent took 7 steps instead of 3. After Langfuse: I see the scratchpad state at each step, latency per tool, and token cost per turn. **Type-annotating each span** (`agent`, `guardrail`, `tool`, `retriever`, `generation`) is the single highest-leverage instrumentation choice I made.

### 5. Evals must be deterministic *first*, LLM-judged *second*

The 13-case eval set started fully LLM-judged and was unreliable across runs. Refactoring to deterministic checks (does the output contain disclaimer? did the diagnosis tool fire? is the top-1 disease in the expected set?) made CI signal trustworthy. Faithfulness/relevancy via LLM-as-judge layers on top, but never gates the build alone.

### 6. RAG quality dominates agent quality

A 100% recall@3 over 41 diseases (FAISS + `bge-small-en-v1.5`) is what makes the diagnosis agent look smart. When I deliberately degraded the index, the orchestrator's reasoning didn't compensate — it just produced confident wrong answers. The lesson: **fix retrieval before tuning prompts.**

### 7. PHI redaction is harder than it looks

Presidio out-of-the-box misses things. Free-text symptoms contain ages, locations, and relational PII ("my 7-year-old daughter") that needed custom recognizers. The redaction also has to be reversible-or-invariant for downstream agents — a naive replacement breaks symptom matching.

### 8. Memory is two different problems

In-session buffer (last N turns, fixed) is mechanical. Cross-session memory via Mem0 is a research problem disguised as a feature: what to write, when to read, how to surface it without poisoning the prompt. I left it as an opt-in for that reason.

---

## Future scope (agentic development)

Roughly ordered by what I'd tackle next.

### Near-term

- **LangGraph variant of the orchestrator** — same tool agents, same guardrails, but a graph-based control flow. Goal: compare debuggability, latency, and token use against the from-scratch loop on the same eval set.
- **MCP server exposing the 4 tool agents** — let the diagnosis/severity/description/precaution agents be consumed by any MCP-compatible client (Claude Desktop, IDE agents, other orchestrators). This forces the tool contracts to stand on their own outside the ReAct loop.
- **HF Spaces deployment** with the Gemini backend, so the system runs without local hardware. The local Gemma4 path stays for offline/private use.

### Mid-term

- **Agent self-critique / verifier loop** — add a verifier agent that scores each candidate `Final Answer` against the scratchpad before it's returned. The interesting question: does verification reduce hallucinations enough to justify the extra latency and tokens?
- **Tool-use planning before action** — instead of one Thought → one Action, generate a plan (sequence of tool calls) and let the orchestrator execute it with re-planning on observation mismatch. This is closer to how production agents (Devin, Manus, Claude Code) actually work.
- **Structured output everywhere** — replace the regex scratchpad parser with a JSON-mode / tool-calling-mode contract on every modern provider. Ship the regex parser as a fallback for local models that don't support it.
- **Eval set expansion to ~100 cases**, with adversarial inputs (prompt injection, PHI bait, off-topic queries, multilingual symptoms) as a separate suite that runs nightly.

### Longer-term / research-flavored

- **Multi-agent debate or council** — two diagnosis agents with different retrieval strategies, plus an arbiter, to test whether ensembling improves faithfulness without nuking latency.
- **Cost-aware routing** — route simple turns to Gemma3-local and complex turns to Gemini, with the routing decision itself logged and evaluated. Build a small classifier on past traces.
- **Reinforcement learning from eval signal** — use the deterministic eval suite as a reward signal to fine-tune a small local policy model that picks tools / writes scratchpad entries. Probably overkill, but a fascinating loop to close.
- **A11y + multimodal input** — symptom photos (rashes, posture), voice input. Each unlocks a new tool agent and a new class of guardrails.
- **Persistent patient memory with consent gates** — Mem0 with explicit per-fact consent prompts, audit log, and a "forget me" path. The interesting design problem is the UI, not the storage.

---

## What I'd do differently next time

- **Start with traces on day one.** I added Langfuse mid-way and had to backfill instrumentation everywhere. Trace-first development would have caught at least three bugs earlier.
- **Write the eval set before the agent.** I wrote the orchestrator first and then designed the eval set to fit it — the right order is the reverse.
- **Pick one provider for the first prototype.** Dual-backend support is great long-term, but it doubled the surface area while I was still learning the basics of tool dispatch.
- **Treat guardrails as a product feature, not a wrapper.** The output policy (disclaimer / emergency banner) is one of the most user-visible parts of the system; it deserved unit tests from the start, not after I shipped.

---

## Status

- 73/73 unit tests passing
- 13/13 agent eval cases passing (deterministic metrics)
- FAISS recall@3 = 100% over 41 diseases
- Langfuse traces flowing end-to-end
- Gradio UI with trace + guard inspection panels
- HF Spaces deployment — pending
- LangGraph variant — pending
- MCP server — pending

---

## Appendix: stack & quick start

| Layer | Technology |
|---|---|
| Agent framework | Custom ReAct orchestrator (Python, ~300 LOC) |
| LLM providers | Google Gemini (cloud) + Gemma3 via Ollama/llama.cpp (local) |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Vector store | FAISS (IndexFlatIP) |
| Memory | In-session buffer + Mem0 cross-session (optional) |
| Guardrails | Presidio (PHI) + rule-based prompt-injection + output policy |
| Observability | Langfuse |
| Evaluation | Custom + LLM-as-judge (RAGAS-style) |
| CI | pytest + GitHub Actions + Promptfoo |
| UI | Gradio (Blocks) |

```bash
# 1. venv + install
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install sentence-transformers faiss-cpu pandas pyyaml \
  presidio-analyzer presidio-anonymizer httpx pydantic python-dotenv \
  google-genai mem0ai langfuse gradio pytest
python -m spacy download en_core_web_lg

# 2. Configure (optional — defaults work for local-only)
cp .env.example .env
# Edit .env to add GOOGLE_API_KEY, LANGFUSE_PUBLIC_KEY+SECRET as needed

# 3. Build the FAISS index (one-time, ~30s)
python scripts/build_index.py

# 4. Pull a local LLM (if going local)
ollama pull gemma3:latest

# 5. Run
python scripts/chat.py                  # CLI REPL
python scripts/ui.py                    # Gradio UI at http://127.0.0.1:7860
python scripts/run_evals.py --no-judge  # Eval suite
python -m pytest tests/                 # Unit tests
```

### Project layout

```
src/medibot/
├── data.py              # CSV loader + alias fixups
├── rag/                 # Embeddings + FAISS store
├── agents/              # 4 tool agents (diagnosis/severity/description/precaution)
├── llm/                 # Provider abstraction (Gemini + Ollama)
├── orchestrator/        # From-scratch ReAct loop + prompts
├── memory/              # Session + Mem0
├── guardrails/          # PHI redaction + injection + output policy
├── observability/       # Langfuse facade
├── evaluation/          # Metrics + LLM-judge + runner
├── ui/                  # Gradio Blocks app
└── app.py               # MediBotApp — top-level API
scripts/                 # chat / ui / build_index / run_evals / promptfoo bridge
tests/                   # pytest unit tests
evals/                   # Golden set + reports
notebooks/               # EDA + RAG build
```
