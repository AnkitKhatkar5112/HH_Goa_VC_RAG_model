"""LLM generation module using Google Gemini.

Generates grounded, context-constrained answers from retrieved passages.
Includes retry with backoff and explicit fallback on timeout.
"""

from __future__ import annotations

import asyncio
import time

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.schemas import GenerationResponse, RetrievedChunk
from app.config import settings


GROUNDED_PROMPT_TEMPLATE = """You are a helpful multilingual assistant. Answer the user's question using ONLY the provided context passages below.

RULES:
1. Use ONLY information from the context passages. Do not use any external knowledge.
2. If the context does not contain enough information to answer the question, say "I don't have enough information in my knowledge base to answer this question."
3. Always cite which passage(s) you used by their IDs (e.g., [Passage 1]).
4. Respond in the SAME LANGUAGE as the user's question.
5. Keep your answer concise and directly relevant.

CONTEXT PASSAGES:
{context}

USER QUESTION: {query}

ANSWER:"""


class GeneratorService:
    """LLM generation service with retry and fallback."""

    def __init__(self):
        self._client = None
        self._initialized = False

    def _get_client(self):
        """Lazy initialization of the Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=settings.google_api_key)
            self._initialized = True
        return self._client

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build the context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Passage {i}] (ID: {chunk.chunk_id}, Language: {chunk.language}):\n{chunk.text}"
            )
        return "\n\n".join(context_parts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def _generate_with_retry(self, prompt: str) -> str:
        """Call the LLM with retry logic."""
        client = self._get_client()
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=prompt,
        )
        return response.text or ""

    async def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        language: str = "en",
    ) -> GenerationResponse:
        """Generate an answer from retrieved context.

        Args:
            query: The user's question.
            chunks: Retrieved passage chunks.
            language: Response language.

        Returns:
            Typed GenerationResponse with answer and citations.
        """
        start = time.perf_counter()

        if not chunks:
            return GenerationResponse(
                answer="I don't have enough information in my knowledge base to answer this question.",
                cited_chunk_ids=[],
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        context = self._build_context(chunks)
        prompt = GROUNDED_PROMPT_TEMPLATE.format(context=context, query=query)

        try:
            # Run LLM call in executor to not block the event loop
            answer = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._generate_with_retry(prompt)
                ),
                timeout=settings.llm_timeout_seconds,
            )

            # Extract cited chunk IDs from the answer
            cited_ids = []
            for i, chunk in enumerate(chunks, 1):
                if f"Passage {i}" in answer or chunk.chunk_id in answer:
                    cited_ids.append(chunk.chunk_id)

            # If no explicit citations found, attribute to all provided chunks
            if not cited_ids:
                cited_ids = [c.chunk_id for c in chunks]

            elapsed_ms = (time.perf_counter() - start) * 1000
            return GenerationResponse(
                answer=answer.strip(),
                cited_chunk_ids=cited_ids,
                latency_ms=round(elapsed_ms, 2),
            )

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Fallback: return retrieved passages directly
            fallback_parts = []
            for i, chunk in enumerate(chunks[:3], 1):
                fallback_parts.append(f"**Passage {i}**: {chunk.text}")
            fallback_answer = (
                "Generation timed out. Here are the most relevant passages:\n\n"
                + "\n\n".join(fallback_parts)
            )
            return GenerationResponse(
                answer=fallback_answer,
                cited_chunk_ids=[c.chunk_id for c in chunks[:3]],
                latency_ms=round(elapsed_ms, 2),
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            fallback_parts = []
            for i, chunk in enumerate(chunks[:3], 1):
                fallback_parts.append(f"**Passage {i}**: {chunk.text}")
            fallback_answer = (
                f"Generation encountered an error. Here are the most relevant passages:\n\n"
                + "\n\n".join(fallback_parts)
            )
            return GenerationResponse(
                answer=fallback_answer,
                cited_chunk_ids=[c.chunk_id for c in chunks[:3]],
                latency_ms=round(elapsed_ms, 2),
            )


# Singleton
generator_service = GeneratorService()
