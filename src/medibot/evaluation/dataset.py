"""Golden-set loader — YAML -> typed Pydantic models.

Parsing is strict-but-friendly: unknown keys become `metadata`, missing
keys get sensible defaults so the YAML can stay terse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class PriorTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Expect(BaseModel):
    blocked: bool | None = None
    tools_called_subset: list[str] = Field(default_factory=list)
    candidates_contain: list[str] = Field(default_factory=list)
    urgency: str | None = None
    emergency_banner: bool | None = None
    disclaimer: bool | None = None
    answer_contains: list[str] = Field(default_factory=list)
    answer_not_contains: list[str] = Field(default_factory=list)
    phi_entities: list[str] = Field(default_factory=list)


class JudgeSpec(BaseModel):
    faithfulness: bool = False
    relevancy: bool = False


class EvalCase(BaseModel):
    id: str
    category: str
    query: str
    setup: list[PriorTurn] = Field(default_factory=list)
    user_id: str | None = None
    expect: Expect = Field(default_factory=Expect)
    judge: JudgeSpec = Field(default_factory=JudgeSpec)

    @property
    def effective_user_id(self) -> str:
        return self.user_id or f"eval_{self.id}"


class EvalSet(BaseModel):
    cases: list[EvalCase]


def load_eval_set(path: str | Path) -> EvalSet:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    return EvalSet.model_validate(raw)
