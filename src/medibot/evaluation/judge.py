"""LLM-as-judge for faithfulness and relevancy.

These are semantic judgements no regex can make. We use our existing
LLMProvider abstraction so the judge can be Gemma4 locally or Gemini
in the cloud via a single env var.

Output: a single integer in [0, 1, 2]
    0 = fails the criterion
    1 = partially meets
    2 = fully meets
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..app import AskResult
from ..llm.base import GenerationConfig, LLMProvider
from .dataset import EvalCase


_FAITHFULNESS_PROMPT = """You are a strict evaluator.

You are given:
- a user question
- the tool observations the assistant received (ground-truth facts it could use)
- the assistant's final answer

Score how FAITHFUL the answer is to the tool observations (0=fails, 1=partially, 2=fully):
- 0: the answer contains factual claims that are NOT supported by any observation (hallucination)
- 1: the answer is mostly grounded but includes minor unsupported claims or asides
- 2: every factual claim in the answer is traceable to an observation OR is a reasonable disclaimer

Respond ONLY with a single JSON object on one line:
{{"score": 0|1|2, "reason": "<one short sentence>"}}

USER QUESTION:
{question}

TOOL OBSERVATIONS (JSON):
{observations}

ASSISTANT FINAL ANSWER:
{answer}

JSON:
"""


_RELEVANCY_PROMPT = """You are a strict evaluator.

You are given a user question and the assistant's final answer.

Score how RELEVANT the answer is to the question (0=fails, 1=partially, 2=fully):
- 0: the answer does not address the question
- 1: the answer partially addresses the question or goes off on tangents
- 2: the answer directly and adequately addresses the question

Respond ONLY with a single JSON object on one line:
{{"score": 0|1|2, "reason": "<one short sentence>"}}

USER QUESTION:
{question}

ASSISTANT FINAL ANSWER:
{answer}

JSON:
"""


@dataclass
class JudgeResult:
    name: str
    score: int  # 0, 1, 2
    reason: str = ""
    ok: bool = False  # True if score >= 1 (passing bar)

    @property
    def mark(self) -> str:
        return "✅" if self.score == 2 else ("⚠️ " if self.score == 1 else "❌")


def _parse_judge_json(text: str) -> tuple[int, str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return (0, f"no JSON found in judge output: {text[:100]}")
    try:
        obj = json.loads(m.group(0))
        score = int(obj.get("score", 0))
        reason = str(obj.get("reason", ""))[:200]
        return (max(0, min(2, score)), reason)
    except Exception as e:  # noqa: BLE001
        return (0, f"parse error: {e}")


def _collect_observations(result: AskResult) -> str:
    """Concatenate all tool observations for this run, as JSON array."""
    if result.run is None:
        return "[]"
    obs = [s.observation for s in result.run.steps if s.observation]
    # Return as a single JSON array so the judge sees structured data
    try:
        parsed = [json.loads(o) for o in obs]
        return json.dumps(parsed, indent=2)
    except Exception:  # noqa: BLE001
        return json.dumps(obs)


def judge_faithfulness(judge_llm: LLMProvider, case: EvalCase, result: AskResult) -> JudgeResult:
    prompt = _FAITHFULNESS_PROMPT.format(
        question=case.query,
        observations=_collect_observations(result),
        answer=result.final_answer or "(empty)",
    )
    out = judge_llm.generate(prompt, config=GenerationConfig(temperature=0.0, max_output_tokens=1024))
    score, reason = _parse_judge_json(out)
    return JudgeResult(name="faithfulness", score=score, reason=reason, ok=(score >= 1))


def judge_relevancy(judge_llm: LLMProvider, case: EvalCase, result: AskResult) -> JudgeResult:
    prompt = _RELEVANCY_PROMPT.format(
        question=case.query,
        answer=result.final_answer or "(empty)",
    )
    out = judge_llm.generate(prompt, config=GenerationConfig(temperature=0.0, max_output_tokens=1024))
    score, reason = _parse_judge_json(out)
    return JudgeResult(name="relevancy", score=score, reason=reason, ok=(score >= 1))


def run_judge(judge_llm: LLMProvider, case: EvalCase, result: AskResult) -> list[JudgeResult]:
    out: list[JudgeResult] = []
    if case.judge.faithfulness:
        out.append(judge_faithfulness(judge_llm, case, result))
    if case.judge.relevancy:
        out.append(judge_relevancy(judge_llm, case, result))
    return out
