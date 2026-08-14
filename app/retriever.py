"""Retrieval wrapper for the FastAPI application.

Thin wrapper around the retrieval module that handles:
- Query embedding at runtime
- FAISS index search
- Language filtering for metadata-aware strategy
- Typed RetrievalResponse output
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from app.schemas import RetrievedChunk, RetrievalResponse
from app.config import settings
from retrieval.embedder import Embedder
from retrieval.indexer import FAISSIndex
from retrieval.chunking import Chunk


class RetrieverService:
    """Manages the embedding model and FAISS index for retrieval."""

    def __init__(self):
        self.embedder: Embedder | None = None
        self.index: FAISSIndex | None = None
        self.chunks_lookup: dict[str, dict] = {}  # chunk_id -> chunk metadata
        self.parent_mapping: dict[str, str] = {}  # chunk_id -> parent text (for parent-doc)
        self._loaded = False

    def load(
        self,
        embedding_model: str | None = None,
    ) -> None:
        """Load the embedding model and build FAISS index dynamically in-memory.

        Args:
            embedding_model: Override embedding model name.
        """
        from retrieval.data_loader import load_msmarco_xi
        from retrieval.chunking import get_chunker

        model_name = embedding_model or settings.embedding_model

        print(f"Loading retriever service (building in-memory index)...")

        # Load embedding model
        self.embedder = Embedder(model_name=model_name)

        # 1. Stream dataset
        print("Streaming dataset...")
        dataset = load_msmarco_xi(
            languages=settings.languages_list,
            sample_size=settings.dataset_sample_size,
        )

        # 2. Chunk
        print("Chunking dataset...")
        chunker = get_chunker(settings.chunk_strategy)
        chunks, pm = chunker.chunk(dataset.passages)
        print(f"Created {len(chunks)} chunks.")

        if pm:
            self.parent_mapping = pm.mapping
            print(f"Created {len(self.parent_mapping)} parent mappings.")

        # 3. Populate lookup
        for c in chunks:
            self.chunks_lookup[c.chunk_id] = {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "language": c.language,
                "passage_id": c.passage_id,
                "strategy": c.strategy,
                "metadata": c.metadata,
            }

        # 4. Embed and Build FAISS
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.embed_passages(chunks)

        print("Building FAISS index...")
        self.index = FAISSIndex(dimension=self.embedder.dimension)
        self.index.build(embeddings, chunks)

        self._loaded = True
        print(f"✓ Retriever service loaded: {self.index.index.ntotal} vectors, dim={self.embedder.dimension}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def retrieve(
        self,
        query: str,
        language: str | None = None,
        top_k: int | None = None,
        use_language_filter: bool = True,
    ) -> RetrievalResponse:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            language: Query language for filtering.
            top_k: Number of results.
            use_language_filter: Whether to apply language filter.

        Returns:
            Typed RetrievalResponse with chunks and timing.
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        k = top_k or settings.top_k
        start = time.perf_counter()

        # Embed query
        import asyncio
        query_emb = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embedder.embed_query(query)
        )

        # Search
        lang_filter = language if use_language_filter else None
        results = self.index.search(query_emb, top_k=k, language_filter=lang_filter)

        # If language filter returned too few results, retry without filter
        if len(results) < k and lang_filter:
            results = self.index.search(query_emb, top_k=k, language_filter=None)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Build response
        chunks = []
        top_score = 0.0
        for r in results:
            chunk_meta = self.chunks_lookup.get(r.chunk_id, {})
            text = chunk_meta.get("text", "")

            # For parent-document strategy, use parent text for context
            if r.chunk_id in self.parent_mapping:
                text = self.parent_mapping[r.chunk_id]

            chunks.append(RetrievedChunk(
                chunk_id=r.chunk_id,
                text=text,
                score=r.score,
                language=chunk_meta.get("language", "unknown"),
                passage_id=chunk_meta.get("passage_id", ""),
            ))
            if r.score > top_score:
                top_score = r.score

        return RetrievalResponse(
            chunks=chunks,
            latency_ms=round(elapsed_ms, 2),
            strategy_used=settings.chunk_strategy,
            top_score=top_score,
        )


# Singleton
retriever_service = RetrieverService()
