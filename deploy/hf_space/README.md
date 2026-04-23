---
title: MediBot
emoji: 🩺
colorFrom: teal
colorTo: indigo
sdk: gradio
sdk_version: "6.13.0"
app_file: app.py
pinned: false
license: mit
short_description: AI multi-agent medical symptom checker (ReAct + FAISS + HF Inference)
---

# MediBot — AI Symptom Checker

Multi-agent medical symptom checker with a from-scratch ReAct orchestrator,
FAISS vector retrieval, Microsoft Presidio PHI redaction, and a Hugging Face
Inference API backend (swap to Google Gemini or local Ollama via env vars).

Source: https://github.com/lkshay/medibot

## Required Space secrets

| Secret | Purpose |
|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | HF Inference API for the LLM calls |
| `LANGFUSE_PUBLIC_KEY` (optional) | Trace every agent turn in Langfuse |
| `LANGFUSE_SECRET_KEY` (optional) | — |
| `LANGFUSE_HOST` (optional) | default `https://cloud.langfuse.com` |

## Tools

1. `diagnosis` — FAISS retrieval over 41 disease documents
2. `severity` — symptom → urgency (low / moderate / urgent / emergency)
3. `description` — plain-language description for a disease
4. `precaution` — preventive / self-care measures

All orchestrated by a custom ReAct loop with Pydantic-typed inputs and a
Langfuse-instrumented span per step.
