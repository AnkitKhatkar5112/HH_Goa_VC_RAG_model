"""FAISS HNSW index for fast in-memory vector search.

Supports building, saving, loading, and querying the index.
Also supports language-filtered retrieval for metadata-aware strategy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import faiss
import numpy as np

from retrieval.chunking import Chunk


@dataclass
class SearchResult:
    """A single search result from the index."""
    chunk_id: str
    score: float
    rank: int


class FAISSIndex:
    """FAISS HNSW index manager."""

    def __init__(
        self,
        dimension: int,
        m: int = 32,
        ef_construction: int = 128,
        ef_search: int = 64,
    ):
        """Initialize the FAISS HNSW index.

        Args:
            dimension: Vector dimension (768 for e5-base, 1024 for e5-large).
            m: Number of neighbors in HNSW graph.
            ef_construction: Construction-time search depth.
            ef_search: Query-time search depth.
        """
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index: faiss.IndexHNSWFlat | None = None
        self.chunk_ids: list[str] = []
        self.chunk_languages: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._lang_to_indices: dict[str, list[int]] = {}

    def build(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Build the HNSW index from embeddings.

        Args:
            embeddings: Embedding matrix of shape (n, dimension).
            chunks: Corresponding chunks (same order as embeddings).
        """
        assert embeddings.shape[0] == len(chunks), "Embeddings and chunks must have same length"
        assert embeddings.shape[1] == self.dimension, f"Expected dim {self.dimension}, got {embeddings.shape[1]}"

        print(f"Building FAISS HNSW index: {len(chunks)} vectors, dim={self.dimension}, M={self.m}")

        self.index = faiss.IndexHNSWFlat(self.dimension, self.m)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        # Normalize for inner product = cosine similarity (embeddings should already be normalized)
        faiss.normalize_L2(embeddings)

        start = time.perf_counter()
        self.index.add(embeddings)
        build_time = time.perf_counter() - start

        # Build ID and language lookup structures
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_languages = [c.language for c in chunks]
        self._id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self._lang_to_indices = {}
        for i, lang in enumerate(self.chunk_languages):
            self._lang_to_indices.setdefault(lang, []).append(i)

        print(f"Index built in {build_time:.2f}s | Total vectors: {self.index.ntotal}")
        print(f"Language distribution: {', '.join(f'{k}: {len(v)}' for k, v in self._lang_to_indices.items())}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        language_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search the index for the nearest neighbors.

        Args:
            query_embedding: Query vector of shape (1, dimension).
            top_k: Number of results to return.
            language_filter: If set, only return results in this language.

        Returns:
            List of SearchResult, sorted by score (descending).
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # If language filter is active, we search with a larger k and filter
        search_k = top_k * 5 if language_filter else top_k

        distances, indices = self.index.search(query_embedding, search_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue

            chunk_id = self.chunk_ids[idx]
            chunk_lang = self.chunk_languages[idx]

            # Apply language filter
            if language_filter and chunk_lang != language_filter:
                continue

            results.append(SearchResult(
                chunk_id=chunk_id,
                score=float(dist),
                rank=len(results),
            ))

            if len(results) >= top_k:
                break

        return results

    def save(self, path: str) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save.")

        os.makedirs(path, exist_ok=True)
        index_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.json")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "m": self.m,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "chunk_ids": self.chunk_ids,
                "chunk_languages": self.chunk_languages,
            }, f, ensure_ascii=False)

        print(f"Saved index ({self.index.ntotal} vectors) and metadata to {path}")

    def load(self, path: str) -> None:
        """Load index and metadata from disk."""
        index_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.json")

        self.index = faiss.read_index(index_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.dimension = meta["dimension"]
        self.m = meta["m"]
        self.ef_construction = meta["ef_construction"]
        self.ef_search = meta["ef_search"]
        self.chunk_ids = meta["chunk_ids"]
        self.chunk_languages = meta["chunk_languages"]
        self._id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self._lang_to_indices = {}
        for i, lang in enumerate(self.chunk_languages):
            self._lang_to_indices.setdefault(lang, []).append(i)

        print(f"Loaded index ({self.index.ntotal} vectors) from {path}")
