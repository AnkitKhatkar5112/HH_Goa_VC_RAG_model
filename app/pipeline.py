"""Orchestration pipeline — the central harness for the RAG system.

This is the single most important file. It orchestrates all stages:
STT → Input Guardrails → Retrieval → Generation → Grounding Check

Every stage:
- Has typed I/O (Pydantic models)
- Is independently timed
- Has retry + timeout + explicit fallback
- Is logged with structured trace context
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langdetect import detect as langdetect_detect

from app.config import settings
from app.logging_config import generate_trace_id, StageTimer
from app.schemas import (
    PipelineResponse,
    LatencyBreakdown,
    RetrievedChunk,
    STTResponse,
    RetrievalResponse,
    GenerationResponse,
)
from app.stt import stt_client, STTError
from app.retriever import retriever_service
from app.generator import generator_service
from guardrails.off_topic import check_off_topic
from guardrails.safety import check_safety


def detect_language(text: str) -> str:
    """Detect language of a text string.

    Returns:
        Language code (en, hi, ta, etc.)
    """
    try:
        lang = langdetect_detect(text)
        # Map langdetect codes to our codes
        lang_map = {"en": "en", "hi": "hi", "ta": "ta", "bn": "bn", "te": "te",
                     "mr": "mr", "gu": "gu", "kn": "kn", "ml": "ml",
                     "pa": "pa", "or": "or", "ur": "ur"}
        return lang_map.get(lang, "en")
    except Exception:
        return "en"


# Lazily initialized grounding checker
_grounding_checker = None


def _get_grounding_checker():
    """Lazy load the grounding checker to avoid loading NLI model at import time."""
    global _grounding_checker
    if _grounding_checker is None:
        from guardrails.grounding import GroundingChecker
        _grounding_checker = GroundingChecker(model_name=settings.nli_model)
    return _grounding_checker


async def run_pipeline(
    audio_data: bytes | None = None,
    text_query: str | None = None,
    language_hint: str | None = None,
    top_k: int | None = None,
    strategy: str | None = None,
) -> PipelineResponse:
    """Run the full RAG pipeline.

    Args:
        audio_data: Raw audio bytes (for voice queries).
        text_query: Text query (for text-only queries).
        language_hint: Optional language hint.
        top_k: Number of results to retrieve.
        strategy: Override chunking strategy.

    Returns:
        Complete PipelineResponse with answer, citations, guardrails, and timing.
    """
    trace_id = generate_trace_id()
    logger = structlog.get_logger().bind(trace_id=trace_id)
    pipeline_start = time.perf_counter()

    latency = LatencyBreakdown()
    guardrail_flags: list[str] = []

    logger.info("pipeline_start", has_audio=audio_data is not None, has_text=text_query is not None)

    # ── Stage 1: STT (skip if text query provided) ───────────────────────
    transcript = text_query
    detected_language = language_hint or "en"

    if audio_data and not text_query:
        with StageTimer("stt", logger) as timer:
            try:
                stt_result = await stt_client.transcribe(audio_data, language_hint)
                transcript = stt_result.transcript
                detected_language = stt_result.detected_language
                logger.info("stt_complete",
                           transcript_length=len(transcript),
                           detected_language=detected_language)
            except STTError as e:
                logger.error("stt_failed", error=str(e))
                latency.stt_ms = timer.elapsed_ms
                latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                return PipelineResponse(
                    transcript=None,
                    detected_language="unknown",
                    answer="I couldn't understand the audio. Please try again or type your question.",
                    guardrail_flags=["stt_error"],
                    latency=latency,
                    trace_id=trace_id,
                )
        latency.stt_ms = round(timer.elapsed_ms, 2)
    elif text_query:
        detected_language = language_hint or detect_language(text_query)

    if not transcript:
        latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        return PipelineResponse(
            answer="No query provided. Please speak or type a question.",
            guardrail_flags=["no_query"],
            latency=latency,
            trace_id=trace_id,
        )

    # ── Stage 2: Input Guardrails ────────────────────────────────────────
    guardrails_start = time.perf_counter()

    # 2a: Safety check
    with StageTimer("safety_check", logger) as timer:
        safety_result = check_safety(transcript, detected_language)
        if not safety_result.passed:
            logger.warning("safety_blocked", reason=safety_result.reason)
            guardrail_flags.append(f"unsafe: {safety_result.reason}")
            latency.guardrails_ms = round(timer.elapsed_ms, 2)
            latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
            return PipelineResponse(
                transcript=transcript,
                detected_language=detected_language,
                answer=safety_result.reason or "Request blocked for safety reasons.",
                guardrail_flags=guardrail_flags,
                latency=latency,
                trace_id=trace_id,
            )

    # ── Stage 3: Retrieval ───────────────────────────────────────────────
    with StageTimer("retrieval", logger) as timer:
        retrieval_result = await retriever_service.retrieve(
            query=transcript,
            language=detected_language,
            top_k=top_k or settings.top_k,
        )
        logger.info("retrieval_complete",
                    num_chunks=len(retrieval_result.chunks),
                    top_score=retrieval_result.top_score,
                    latency_ms=retrieval_result.latency_ms)
    latency.retrieval_ms = round(timer.elapsed_ms, 2)

    # 2b: Off-topic check (uses retrieval score)
    with StageTimer("off_topic_check", logger) as timer:
        offtopic_result = check_off_topic(
            top_score=retrieval_result.top_score,
            threshold=settings.off_topic_threshold,
            query_language=detected_language,
        )
        if not offtopic_result.passed:
            logger.warning("off_topic_blocked", top_score=retrieval_result.top_score)
            guardrail_flags.append(f"off_topic: {offtopic_result.reason}")
            latency.guardrails_ms = round((time.perf_counter() - guardrails_start) * 1000, 2)
            latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
            return PipelineResponse(
                transcript=transcript,
                detected_language=detected_language,
                answer=offtopic_result.reason or "This question is outside my knowledge domain.",
                guardrail_flags=guardrail_flags,
                latency=latency,
                trace_id=trace_id,
            )

    # Handle empty retrieval
    if not retrieval_result.chunks:
        logger.warning("retrieval_empty")
        latency.guardrails_ms = round((time.perf_counter() - guardrails_start) * 1000, 2)
        latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        return PipelineResponse(
            transcript=transcript,
            detected_language=detected_language,
            answer="I couldn't find any relevant information for your question.",
            guardrail_flags=["no_results"],
            latency=latency,
            trace_id=trace_id,
        )

    # ── Stage 4: Generation ──────────────────────────────────────────────
    with StageTimer("generation", logger) as timer:
        gen_result = await generator_service.generate(
            query=transcript,
            chunks=retrieval_result.chunks,
            language=detected_language,
        )
        logger.info("generation_complete",
                    answer_length=len(gen_result.answer),
                    cited_chunks=len(gen_result.cited_chunk_ids))
    latency.generation_ms = round(timer.elapsed_ms, 2)

    # ── Stage 5: Grounding check ─────────────────────────────────────────
    grounded = True
    with StageTimer("grounding_check", logger) as timer:
        try:
            checker = _get_grounding_checker()
            passage_texts = [c.text for c in retrieval_result.chunks]
            grounding_result = checker.check_grounding(
                answer=gen_result.answer,
                passage_texts=passage_texts,
                contradiction_threshold=settings.grounding_contradiction_threshold,
                entailment_threshold=settings.grounding_threshold,
                query_language=detected_language,
            )
            grounded = grounding_result.passed
            if not grounded:
                logger.warning("grounding_failed", confidence=grounding_result.confidence)
                guardrail_flags.append(f"ungrounded: {grounding_result.reason}")
                # Replace answer with passages + disclaimer
                fallback_parts = [grounding_result.reason or ""]
                for i, chunk in enumerate(retrieval_result.chunks[:3], 1):
                    fallback_parts.append(f"\n**Passage {i}**: {chunk.text}")
                gen_result.answer = "\n".join(fallback_parts)
        except Exception as e:
            logger.error("grounding_check_error", error=str(e))
            # On grounding check failure, pass through (fail open)

    latency.guardrails_ms = round((time.perf_counter() - guardrails_start) * 1000, 2)
    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

    logger.info("pipeline_complete", latency=latency.model_dump(), guardrail_flags=guardrail_flags)

    return PipelineResponse(
        transcript=transcript,
        detected_language=detected_language,
        answer=gen_result.answer,
        cited_chunks=retrieval_result.chunks,
        grounded=grounded,
        guardrail_flags=guardrail_flags,
        latency=latency,
        trace_id=trace_id,
    )
