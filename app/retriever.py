"""Retrieval wrapper for the FastAPI application.

Thin wrapper around the retrieval module that handles:
- Query embedding at runtime
- ChromaDB persistent index search
- Language and metadata filtering
- Routing across multiple chunking strategies
- Typed RetrievalResponse output
"""

from __future__ import annotations

import os
import time

from app.schemas import RetrievedChunk, RetrievalResponse
from app.config import settings
from retrieval.embedder import Embedder
from retrieval.indexer import ChromaIndex


class RetrieverService:
    """Manages the embedding model and ChromaDB index for retrieval."""

    def __init__(self):
        self.embedder: Embedder | None = None
        self.index: ChromaIndex | None = None
        self._loaded = False
        self.strategies = ["fixed_size", "semantic", "metadata_aware"]

    def load(
        self,
        embedding_model: str | None = None,
    ) -> None:
        """Load the embedding model and connect to existing ChromaDB index.

        Args:
            embedding_model: Override embedding model name.
            
        Raises:
            RuntimeError: If the ChromaDB directory does not exist or collections are empty.
        """
        model_name = embedding_model or settings.embedding_model

        print(f"Loading retriever service with ChromaDB...")

        # Verify DB exists
        if not os.path.exists(settings.chroma_db_dir):
            raise RuntimeError(
                f"ChromaDB not found at {settings.chroma_db_dir}. "
                f"You must run 'python scripts/build_index.py' before starting the server."
            )

        # Load embedding model
        self.embedder = Embedder(model_name=model_name)
        
        # Connect to existing ChromaDB
        self.index = ChromaIndex(
            persist_directory=settings.chroma_db_dir,
            dimension=self.embedder.dimension
        )

        # Verify collections
        for strategy in self.strategies:
            try:
                # This will raise an exception or return an empty collection if it doesn't exist
                collection = self.index.client.get_collection(name=strategy)
                count = collection.count()
                if count == 0:
                     raise ValueError(f"Collection '{strategy}' is empty.")
                self.index.collections[strategy] = collection
                print(f"✓ Verified collection '{strategy}' (vectors: {count})")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load collection '{strategy}': {e}. "
                    f"Please re-run 'python scripts/build_index.py'."
                )

        # Warmup: force PyTorch JIT compilation + pre-load ChromaDB HNSW indexes
        print("Warming up embedding model + ChromaDB indexes...")
        warmup_emb = self.embedder.embed_query("warmup query")
        for strategy in self.strategies:
            try:
                self.index.search(warmup_emb, collection_name=strategy, top_k=1)
            except Exception:
                pass
        print("✓ Embedding model + ChromaDB indexes warm")

        self._loaded = True
        print(f"✓ Retriever service ready. Connected to ChromaDB at: {settings.chroma_db_dir}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def retrieve(
        self,
        query: str,
        language: str | None = None,
        top_k: int | None = None,
        use_language_filter: bool = True,
        strategy: str | None = None,
    ) -> RetrievalResponse:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            language: Query language for filtering.
            top_k: Number of results.
            use_language_filter: Whether to apply language filter.
            strategy: Specify strategy, or None to route across all and pick best.

        Returns:
            Typed RetrievalResponse with chunks and timing.
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        k = top_k or settings.top_k
        start = time.perf_counter()

        # Embed query
        import asyncio
        embed_start = time.perf_counter()
        query_emb = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embedder.embed_query(query)
        )
        embed_ms = (time.perf_counter() - embed_start) * 1000

        # Skip expensive ChromaDB where-clause filter when only one language is indexed.
        # ChromaDB metadata filtering adds ~250ms per query — with 3 collections that's
        # ~750ms of pure overhead when all data is already in the target language.
        single_language = len(settings.languages_list) == 1
        lang_filter = None if single_language else (language if use_language_filter else None)
        
        target_strategies = [strategy] if strategy else self.strategies
        all_results = []

        # Query all target collections
        search_start = time.perf_counter()
        for s in target_strategies:
            # Only apply is_selected filter for multi-language setups where precision matters
            is_selected_filter = True if (s == "metadata_aware" and not single_language) else None
            try:
                results = self.index.search(
                    query_emb,
                    collection_name=s,
                    top_k=k,
                    language_filter=lang_filter,
                    is_selected_filter=is_selected_filter
                )
                
                # If language filter returned too few results, retry without filter
                if len(results) < k and lang_filter:
                    results = self.index.search(
                        query_emb,
                        collection_name=s,
                        top_k=k,
                        language_filter=None,
                        is_selected_filter=is_selected_filter
                    )
                    
                all_results.extend(results)
            except Exception as e:
                print(f"Error querying strategy {s}: {e}")
        search_ms = (time.perf_counter() - search_start) * 1000

        # Sort and pick top k across all strategies
        all_results.sort(key=lambda x: x.score, reverse=True)
        best_results = all_results[:k]

        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[Retrieval timing] embed={embed_ms:.1f}ms search={search_ms:.1f}ms total={elapsed_ms:.1f}ms")

        # Build response
        chunks = []
        top_score = 0.0
        used_strategies = set()
        
        for r in best_results:
            used_strategies.add(r.strategy_name)
            chunks.append(RetrievedChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                language=r.language,
                passage_id=r.passage_id,
            ))
            if r.score > top_score:
                top_score = r.score
                
        reported_strategy = strategy if strategy else ",".join(list(used_strategies))

        return RetrievalResponse(
            chunks=chunks,
            latency_ms=round(elapsed_ms, 2),
            strategy_used=reported_strategy,
            top_score=top_score,
        )


# Singleton
retriever_service = RetrieverService()
