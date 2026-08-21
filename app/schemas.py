"""Typed Pydantic schemas for every stage of the RAG pipeline.

Every stage boundary has typed I/O — no free strings passed around.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# STT Stage
# ─────────────────────────────────────────────────────────────────────────────

class STTResponse(BaseModel):
    """Output of the Speech-to-Text stage."""
    transcript: str
    detected_language: str
    confidence: float = 0.0
    latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Stage
# ─────────────────────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single retrieved passage chunk."""
    chunk_id: str
    text: str
    score: float
    language: str
    passage_id: str


class RetrievalResponse(BaseModel):
    """Output of the retrieval stage."""
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    strategy_used: str = ""
    top_score: float = 0.0  # Highest retrieval score (for off-topic check)


# ─────────────────────────────────────────────────────────────────────────────
# Generation Stage
# ─────────────────────────────────────────────────────────────────────────────

class GenerationResponse(BaseModel):
    """Output of the LLM generation stage."""
    answer: str
    source: str = "generated"
    cited_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Latency Breakdown
# ─────────────────────────────────────────────────────────────────────────────

class LatencyBreakdown(BaseModel):
    """Stage-by-stage latency for a single request."""
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    guardrails_ms: float = 0.0
    total_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Response
# ─────────────────────────────────────────────────────────────────────────────

class PipelineResponse(BaseModel):
    """Complete response from the RAG pipeline."""
    transcript: str | None = None
    detected_language: str = "en"
    answer: str = ""
    source: str = "generated"
    cited_chunks: list[RetrievedChunk] = Field(default_factory=list)
    grounded: bool = True
    guardrail_flags: list[str] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    trace_id: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# API Request Models
# ─────────────────────────────────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    """Request body for text-only queries."""
    query: str
    language: str | None = None
    top_k: int = 5
    strategy: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "1.0.0"
    index_loaded: bool = False
    languages: list[str] = Field(default_factory=list)
    total_vectors: int = 0
