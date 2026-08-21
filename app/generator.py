"""LLM generation module using Groq API.

Generates grounded, context-constrained answers from retrieved passages.
Includes retry with backoff and explicit fallback on timeout.
"""

from __future__ import annotations

import asyncio
import time
import os

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.schemas import GenerationResponse, RetrievedChunk
from app.config import settings

from groq import Groq


GROUNDED_PROMPT_TEMPLATE = """You are a helpful multilingual assistant. Answer the user's question using ONLY the provided context passages below.

RULES:
1. Use ONLY information from the context passages. Do not use any external knowledge.
2. If the context does not contain enough information to answer the question, say "I don't have enough information in my knowledge base to answer this question."
3. Always cite which passage(s) you used by their IDs (e.g., [Passage 1]).
4. Respond in the SAME LANGUAGE as the user's question.
5. Keep your answer concise and directly relevant.
6. Restrict your answer to a maximum of 3 sentences or about 150 words.

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
        """Lazy initialization of the Groq client."""
        if self._client is None:
            api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY")
            self._client = Groq(api_key=api_key)
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

    def _generate_with_retry(self, prompt: str) -> tuple[str, int, dict]:
        """Call the LLM with retry logic. Returns (answer, retries, headers)."""
        client = self._get_client()
        max_retries = settings.llm_max_retries
        retries = 0
        for attempt in range(max_retries):
            try:
                raw_response = client.chat.completions.with_raw_response.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=settings.llm_model,
                    max_tokens=600,
                    extra_body={"reasoning_effort": "low"},
                )
                
                headers = raw_response.headers
                req_rem = headers.get('x-ratelimit-remaining-requests', 'unknown')
                tok_rem = headers.get('x-ratelimit-remaining-tokens', 'unknown')
                
                print(f"[LLM Call] Attempt {attempt+1}: model={settings.llm_model}, remaining_reqs={req_rem}, remaining_tokens={tok_rem}")
                
                parsed = raw_response.parse()
                msg = parsed.choices[0].message
                reasoning = getattr(msg, "reasoning", getattr(msg, "reasoning_content", None))
                if reasoning:
                    print(f"[LLM Call] Captured Reasoning: {reasoning}")
                else:
                    print(f"[LLM Call] No reasoning captured or field empty.")
                    
                return msg.content or "", retries, dict(headers)
                
            except Exception as e:
                retries += 1
                print(f"[LLM Call] Attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return "", retries, {}

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
            answer, retries, headers = await asyncio.wait_for(
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
                source="fallback",
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
                source="fallback",
                cited_chunk_ids=[c.chunk_id for c in chunks[:3]],
                latency_ms=round(elapsed_ms, 2),
            )


# Singleton
generator_service = GeneratorService()
