#!/usr/bin/env python
"""Run the MediBot eval suite.

Usage:
    python scripts/run_evals.py                               # full suite, verbose
    python scripts/run_evals.py --category safety             # filter by category
    python scripts/run_evals.py --judge-model gemma4:31b      # use bigger judge
    python scripts/run_evals.py --no-judge                    # skip LLM-as-judge
    python scripts/run_evals.py --output evals/reports/run.json
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    p = argparse.ArgumentParser(description="Run the MediBot eval suite.")
    p.add_argument("--set", default="evals/eval_set.yaml", help="Path to eval set YAML.")
    p.add_argument("--llm", default="ollama", choices=["ollama", "gemini"], help="App LLM backend.")
    p.add_argument("--model", default=None, help="Override app LLM model name.")
    p.add_argument("--judge-provider", default=None, help="Judge LLM backend (defaults to --llm).")
    p.add_argument("--judge-model", default=None, help="Override judge LLM model (e.g. gemma4:31b).")
    p.add_argument("--no-judge", action="store_true", help="Disable LLM-as-judge metrics.")
    p.add_argument("--category", default=None, help="Filter to a single category.")
    p.add_argument("--output", default="evals/reports/latest.json", help="Write JSON report.")
    p.add_argument("--quiet", action="store_true", help="Only print summary.")
    args = p.parse_args()

    if args.model:
        if args.llm == "ollama":
            os.environ["OLLAMA_MODEL"] = args.model
        else:
            os.environ["GEMINI_MODEL"] = args.model

    from medibot.evaluation import run_eval_suite

    report = run_eval_suite(
        eval_set_path=args.set,
        llm_provider=args.llm,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        filter_category=args.category,
        output_json=args.output,
        verbose=(not args.quiet),
        disable_judge=args.no_judge,
    )

    report.print_summary()
    report.print_failures_detail()

    return 0 if report.pass_count == report.total else 1


if __name__ == "__main__":
    sys.exit(main())
