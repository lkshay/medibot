"""A from-scratch ReAct orchestrator.

Why from scratch: to teach the Reason-Act-Observe loop explicitly. A LangGraph
version ships alongside in `langgraph_app.py` for the production stack.

Design:
    - We keep a growing "scratchpad" string of Thought/Action/Action Input/Observation
    - Each LLM call is `system + (prior turns) + scratchpad` with stop=["\\nObservation:"]
    - We parse the LLM output, dispatch the tool via our Tool registry, append
      the observation, and loop until Final Answer or max_iters
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from ..llm.base import GenerationConfig, LLMProvider
from ..observability import get_tracer
from .prompts import SYSTEM_PROMPT


# ============================================================================
# Tool registration
# ============================================================================
@dataclass
class Tool:
    """A tool the orchestrator can call."""

    name: str
    description: str
    input_model: type[BaseModel]
    call: Callable[[BaseModel], BaseModel]

    def signature(self) -> str:
        """Render a compact signature for the system prompt."""
        schema = self.input_model.model_json_schema()
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        lines = []
        for pname, spec in props.items():
            # resolve $ref if present (Pydantic sometimes uses these)
            ptype = spec.get("type") or spec.get("anyOf", [{}])[0].get("type", "any")
            if ptype == "array":
                item_type = spec.get("items", {}).get("type", "any")
                ptype = f"array<{item_type}>"
            marker = "" if pname in required else " (optional)"
            desc = spec.get("description", "").replace("\n", " ")
            lines.append(f'  "{pname}": {ptype}{marker}  // {desc}')
        body = "\n".join(lines) if lines else "  (no arguments)"
        return f"{self.name}({{\n{body}\n}})\n  {self.description}"


# ============================================================================
# Run record
# ============================================================================
@dataclass
class ReactStep:
    thought: str = ""
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    final_answer: str | None = None
    raw: str = ""


@dataclass
class ReactRun:
    query: str
    steps: list[ReactStep] = field(default_factory=list)
    final_answer: str | None = None
    iterations: int = 0
    terminated_reason: str = ""

    def transcript(self) -> str:
        """A human-readable trace, useful for logs / notebooks."""
        lines = [f"User: {self.query}", ""]
        for i, s in enumerate(self.steps, 1):
            lines.append(f"--- step {i} ---")
            lines.append(f"Thought: {s.thought}")
            if s.action:
                lines.append(f"Action: {s.action}")
                lines.append(f"Action Input: {json.dumps(s.action_input)}")
                lines.append(f"Observation: {s.observation}")
            if s.final_answer:
                lines.append(f"Final Answer: {s.final_answer}")
        lines.append(f"\n[terminated: {self.terminated_reason} in {self.iterations} step(s)]")
        return "\n".join(lines)


# ============================================================================
# Orchestrator
# ============================================================================
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class ReactOrchestrator:
    def __init__(
        self,
        llm: LLMProvider,
        tools: list[Tool],
        max_iters: int = 6,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> None:
        self.llm = llm
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        self.max_iters = max_iters
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    # ---------- prompt assembly --------------------------------------------------
    def _system_prompt(self) -> str:
        tool_blob = "\n\n".join(t.signature() for t in self.tools.values())
        return SYSTEM_PROMPT.format(tool_descriptions=tool_blob)

    def _build_user_turn(self, query: str, history: list[dict] | None) -> str:
        """Render prior turns (if any) plus the current question."""
        parts: list[str] = []
        if history:
            parts.append("Prior conversation:")
            for turn in history:
                parts.append(f"{turn['role'].title()}: {turn['content']}")
            parts.append("")
        parts.append(f"Current question: {query}")
        parts.append("")  # scratchpad slot appended by caller
        return "\n".join(parts)

    # ---------- parsing ----------------------------------------------------------
    @staticmethod
    def _extract_json(block: str) -> dict | None:
        """Pull a JSON object out of text. Tolerates ```json fences and stray prose."""
        m = _FENCED.search(block) or _JSON_OBJ.search(block)
        if not m:
            return None
        try:
            return json.loads(m.group(1) if m.lastindex else m.group(0))
        except json.JSONDecodeError:
            return None

    def _parse(self, text: str) -> ReactStep:
        step = ReactStep(raw=text)

        # Final Answer short-circuits
        if "Final Answer:" in text:
            thought = re.search(r"Thought:\s*(.+?)(?=\n\s*Final Answer:)", text, re.DOTALL)
            step.thought = thought.group(1).strip() if thought else ""
            step.final_answer = text.split("Final Answer:", 1)[1].strip()
            return step

        thought = re.search(r"Thought:\s*(.+?)(?=\n\s*Action:)", text, re.DOTALL)
        action = re.search(r"Action:\s*([\w\-]+)", text)
        ainput = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)

        step.thought = thought.group(1).strip() if thought else text.strip()
        if action:
            step.action = action.group(1).strip()
        if ainput:
            step.action_input = self._extract_json(ainput.group(1).strip())
        return step

    # ---------- tool dispatch ----------------------------------------------------
    def _dispatch(self, name: str, args: dict | None) -> str:
        tracer = get_tracer()
        tool = self.tools.get(name)
        if tool is None:
            err = json.dumps({"error": f"Unknown tool {name!r}. Available: {list(self.tools)}"})
            with tracer.span(name=f"tool:{name}", as_type="tool", input=args) as ts:
                ts.update(output={"error": "unknown_tool"}, level="WARNING")
            return err

        with tracer.span(name=f"tool:{name}", as_type="tool", input=args) as tool_span:
            try:
                validated = tool.input_model(**(args or {}))
            except ValidationError as e:
                err = json.dumps({"error": "validation_failed", "details": e.errors()})
                tool_span.update(output={"error": "validation_failed"}, level="WARNING")
                return err
            try:
                result = tool.call(validated)
            except Exception as e:  # noqa: BLE001
                err = json.dumps({"error": f"{type(e).__name__}: {e}"})
                tool_span.update(output={"error": str(e)}, level="ERROR")
                return err
            payload = result.model_dump_json() if isinstance(result, BaseModel) else json.dumps(result)
            tool_span.update(
                output=(result.model_dump() if isinstance(result, BaseModel) else result),
            )
            return payload

    # ---------- main loop --------------------------------------------------------
    def run(self, query: str, history: list[dict] | None = None) -> ReactRun:
        tracer = get_tracer()
        run = ReactRun(query=query)
        turn_prefix = self._build_user_turn(query, history)
        scratchpad = ""

        with tracer.span(
            name="react_loop",
            as_type="chain",
            input={"query": query, "history_turns": len(history or [])},
        ) as loop_span:
            for i in range(self.max_iters):
                run.iterations = i + 1
                prompt = turn_prefix + scratchpad

                with tracer.span(name=f"react_step_{i + 1}", as_type="span") as step_span:
                    with tracer.generation(
                        name="llm.generate",
                        model=self.llm.name,
                        input=prompt,
                        metadata={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_output_tokens,
                        },
                    ) as gen:
                        output = self.llm.generate(
                            prompt=prompt,
                            system=self._system_prompt(),
                            config=GenerationConfig(
                                temperature=self.temperature,
                                max_output_tokens=self.max_output_tokens,
                                stop=["\nObservation:", "\nObservation :"],
                            ),
                        ).strip()
                        # Feed token counts to Langfuse if the provider captured them.
                        usage = getattr(self.llm, "last_usage", None)
                        if usage:
                            gen.update(output=output, usage_details=usage)
                        else:
                            gen.update(output=output)

                    step = self._parse(output)
                    run.steps.append(step)

                    # Terminal: Final Answer
                    if step.final_answer is not None:
                        run.final_answer = step.final_answer
                        run.terminated_reason = "completed"
                        step_span.update(
                            output={"thought": step.thought[:200], "final_answer": step.final_answer[:200]}
                        )
                        loop_span.update(
                            output={
                                "iterations": run.iterations,
                                "terminated": "completed",
                                "final_preview": step.final_answer[:200],
                            }
                        )
                        return run

                    # No action parsed — give up gracefully
                    if step.action is None:
                        run.terminated_reason = "no_action_parsed"
                        step_span.update(output={"error": "no_action_parsed"}, level="WARNING")
                        loop_span.update(
                            output={"iterations": run.iterations, "terminated": "no_action_parsed"},
                            level="WARNING",
                        )
                        return run

                    # Execute tool (this opens its own nested span)
                    observation = self._dispatch(step.action, step.action_input)
                    step.observation = observation
                    step_span.update(
                        output={
                            "thought": step.thought[:200],
                            "action": step.action,
                            "action_input": step.action_input,
                        }
                    )

                    # Extend the scratchpad
                    scratchpad += (
                        f"Thought: {step.thought}\n"
                        f"Action: {step.action}\n"
                        f"Action Input: {json.dumps(step.action_input or {})}\n"
                        f"Observation: {observation}\n"
                    )

            run.terminated_reason = "max_iters"
            loop_span.update(
                output={"iterations": run.iterations, "terminated": "max_iters"},
                level="WARNING",
            )
            return run
