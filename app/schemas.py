"""Pydantic schemas shared across the LifeOps agent."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

IntentType = Literal[
    "reminder", "calendar_event", "journal_entry", "daily_summary", "unknown"
]
Priority = Literal["low", "medium", "high"]


class ParsedIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_type: IntentType
    title: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    priority: Optional[Priority] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    raw_text: str


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict
    status: Literal["success", "error"]
    result: Optional[dict] = None
    error: Optional[str] = None


class AgentTurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    done: bool
    message: str
    stored_record: Optional[dict] = None
    tool_called: Optional[str] = None
    actions: list[ToolAction] = []
    trace_id: Optional[str] = None
