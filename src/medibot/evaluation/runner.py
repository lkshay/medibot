"""Eval runner — loads the golden set, runs each case, scores, reports."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..app import AskResult, MediBotApp
from ..llm import get_llm
from ..llm.base import LLMProvider
from .dataset import EvalCase, EvalSet, load_eval_set
from .judge import JudgeResult, run_judge
from .metrics import MetricResult, run_all_metrics


@dataclass
class CaseReport:
    case_id: str
    category: str
    query: str
    final_answer: str
    blocked: bool
    iterations: int
    latency_s: float
    metrics: list[MetricResult]
    judge_results: list[JudgeResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics) and all(j.ok for j in self.judge_results)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "query": self.query,
            "final_answer": self.final_answer[:500],
            "blocked": self.blocked,
            "iterations": self.iterations,
            "latency_s": round(self.latency_s, 2),
            "passed": self.passed,
            "metrics": [asdict(m) for m in self.metrics],
            "judge": [asdict(j) for j in self.judge_results],
        }


@dataclass
class RunReport:
    cases: list[CaseReport]
    elapsed_s: float
    judge_model: str | None

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def total(self) -> int:
        return len(self.cases)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total": self.total,
                "passed": self.pass_count,
                "failed": self.total - self.pass_count,
                "pass_rate": round(self.pass_count / self.total, 3) if self.total else None,
                "elapsed_s": round(self.elapsed_s, 1),
                "judge_model": self.judge_model,
            },
            "cases": [c.to_dict() for c in self.cases],
        }

    def print_summary(self) -> None:
        print()
        print("=" * 72)
        print(f"{'Case':<36} {'Cat':<10} {'Metrics':<22} {'Judge':<10} {'Time'}")
        print("-" * 72)
        for c in self.cases:
            metric_marks = "".join(m.mark for m in c.metrics)
            judge_marks = "".join(j.mark for j in c.judge_results) if c.judge_results else "-"
            print(f"{c.case_id:<36} {c.category:<10} {metric_marks:<22} {judge_marks:<10} {c.latency_s:.1f}s")
        print("-" * 72)
        rate = (self.pass_count / self.total * 100) if self.total else 0
        print(f"TOTAL  {self.pass_count}/{self.total} passed  ({rate:.0f}%)   elapsed: {self.elapsed_s:.1f}s")
        print("=" * 72)

    def print_failures_detail(self) -> None:
        failures = [c for c in self.cases if not c.passed]
        if not failures:
            return
        print()
        print("Failure details:")
        for c in failures:
            print(f"\n  [{c.case_id}]  {c.query}")
            for m in c.metrics:
                if not m.passed:
                    print(f"    ❌ {m.name}: {m.detail}")
            for j in c.judge_results:
                if not j.ok:
                    print(f"    ❌ judge.{j.name}: score={j.score} {j.reason}")


def _warm_memory(app: MediBotApp, case: EvalCase) -> None:
    """Replay any prior turns into session memory before running the target query."""
    if not case.setup:
        return
    conv = app.session(case.effective_user_id)
    for turn in case.setup:
        conv.add_turn(turn.role, turn.content)


def run_case(app: MediBotApp, case: EvalCase, judge_llm: LLMProvider | None) -> CaseReport:
    app.reset_session(case.effective_user_id)
    _warm_memory(app, case)

    t0 = time.time()
    result = app.ask(case.effective_user_id, case.query)
    elapsed = time.time() - t0

    metrics = run_all_metrics(case, result)
    judge_results: list[JudgeResult] = []
    if judge_llm is not None and (case.judge.faithfulness or case.judge.relevancy):
        judge_results = run_judge(judge_llm, case, result)

    return CaseReport(
        case_id=case.id,
        category=case.category,
        query=case.query,
        final_answer=result.final_answer or "",
        blocked=result.blocked,
        iterations=result.run.iterations if result.run else 0,
        latency_s=elapsed,
        metrics=metrics,
        judge_results=judge_results,
    )


def run_eval_suite(
    eval_set_path: str | Path = "evals/eval_set.yaml",
    llm_provider: str | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    filter_category: str | None = None,
    output_json: str | Path | None = None,
    verbose: bool = False,
    disable_judge: bool = False,
) -> RunReport:
    eval_set = load_eval_set(eval_set_path)
    cases = eval_set.cases
    if filter_category:
        cases = [c for c in cases if c.category == filter_category]
    if disable_judge:
        for c in cases:
            c.judge.faithfulness = False
            c.judge.relevancy = False

    app = MediBotApp.build(llm_provider=llm_provider)

    # Optional LLM-as-judge
    judge_llm: LLMProvider | None = None
    if any(c.judge.faithfulness or c.judge.relevancy for c in cases):
        import os as _os
        if judge_model:
            _os.environ["OLLAMA_MODEL"] = judge_model
            _os.environ["GEMINI_MODEL"] = judge_model
        judge_llm = get_llm(judge_provider or llm_provider)
        print(f"[judge] using {judge_llm.name}")

    reports: list[CaseReport] = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        if verbose:
            print(f"[{i}/{len(cases)}] {case.id}  ({case.category})  {case.query[:60]}...")
        report = run_case(app, case, judge_llm)
        reports.append(report)
        if verbose:
            print(f"    -> {'PASS' if report.passed else 'FAIL'}  {report.latency_s:.1f}s  steps={report.iterations}")
    elapsed = time.time() - t0

    run_report = RunReport(
        cases=reports,
        elapsed_s=elapsed,
        judge_model=(judge_llm.name if judge_llm else None),
    )

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(run_report.to_dict(), indent=2))

    return run_report
