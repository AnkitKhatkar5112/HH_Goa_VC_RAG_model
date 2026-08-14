"""Typed schemas for guardrail check results."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


class GuardrailResult(BaseModel):
    """Result of a guardrail check."""
    passed: bool
    reason: str | None = None
    category: Literal["off_topic", "unsafe", "hallucination"]
    confidence: float = 0.0


class GuardrailSummary(BaseModel):
    """Aggregate guardrail results for a single query."""
    all_passed: bool
    flags: list[GuardrailResult]
    refusal_message: str | None = None
