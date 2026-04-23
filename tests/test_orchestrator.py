"""Unit tests for the ReAct orchestrator — parser and tool-dispatch only.

The full run() loop needs an LLM and is covered by the eval suite instead.
"""

import json

from pydantic import BaseModel

from medibot.llm.base import GenerationConfig, LLMProvider, Message
from medibot.orchestrator import ReactOrchestrator, Tool


class _EchoInput(BaseModel):
    text: str


class _EchoOutput(BaseModel):
    echo: str


def _echo_tool_fn(inp: _EchoInput) -> _EchoOutput:
    return _EchoOutput(echo=inp.text.upper())


class _NullLLM(LLMProvider):
    """No-network LLM stub — orchestrator parse/dispatch tests only."""

    @property
    def name(self) -> str:
        return "test:null"

    def generate(self, prompt: str, system: str = "", config: GenerationConfig | None = None) -> str:
        return ""

    def chat(self, messages: list[Message], config: GenerationConfig | None = None) -> str:
        return ""


def _make_orchestrator() -> ReactOrchestrator:
    tool = Tool(
        name="echo",
        description="Echo uppercased text.",
        input_model=_EchoInput,
        call=_echo_tool_fn,
    )
    return ReactOrchestrator(llm=_NullLLM(), tools=[tool], max_iters=3)


class TestReactParser:
    def test_parses_thought_action_input(self):
        orch = _make_orchestrator()
        text = (
            "Thought: I need to echo the word.\n"
            "Action: echo\n"
            'Action Input: {"text": "hello"}'
        )
        step = orch._parse(text)
        assert step.action == "echo"
        assert step.action_input == {"text": "hello"}
        assert "echo the word" in step.thought

    def test_parses_final_answer(self):
        orch = _make_orchestrator()
        text = (
            "Thought: I have enough to answer.\n"
            "Final Answer: The capital of France is Paris."
        )
        step = orch._parse(text)
        assert step.final_answer == "The capital of France is Paris."
        assert step.action is None

    def test_parses_json_in_code_fence(self):
        orch = _make_orchestrator()
        text = (
            "Thought: fenced json case.\n"
            "Action: echo\n"
            "Action Input: ```json\n"
            '{"text": "fenced"}\n'
            "```"
        )
        step = orch._parse(text)
        assert step.action_input == {"text": "fenced"}

    def test_parse_returns_empty_action_on_malformed(self):
        orch = _make_orchestrator()
        text = "Thought: no action here, no action input either."
        step = orch._parse(text)
        assert step.action is None


class TestToolDispatch:
    def test_valid_call(self):
        orch = _make_orchestrator()
        out = orch._dispatch("echo", {"text": "hi"})
        parsed = json.loads(out)
        assert parsed == {"echo": "HI"}

    def test_unknown_tool(self):
        orch = _make_orchestrator()
        out = orch._dispatch("does_not_exist", {"text": "x"})
        parsed = json.loads(out)
        assert "error" in parsed

    def test_validation_error(self):
        orch = _make_orchestrator()
        # missing required "text" field
        out = orch._dispatch("echo", {})
        parsed = json.loads(out)
        assert parsed.get("error") == "validation_failed"
