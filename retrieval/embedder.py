"""Embedding module using multilingual-e5 models.

Handles batch encoding of passages and queries with proper prefixing
and L2 normalization for cosine similarity.
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retrieval.chunking import Chunk


class Embedder:
    """Wrapper around multilingual-e5 sentence transformer."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str | None = None):
        """Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to run inference on ("cpu", "cuda", or None for auto).
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {model_name} (dim={self.dimension})")

    def embed_passages(
        self,
        chunks: list[Chunk],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed passage chunks with 'passage: ' prefix.

        Args:
            chunks: List of Chunk objects to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            Normalized embedding matrix of shape (n_chunks, dimension).
        """
        texts = [f"passage: {chunk.text}" for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query with 'query: ' prefix.

        Args:
            query: The search query string.

        Returns:
            Normalized embedding vector of shape (1, dimension).
        """
        text = f"query: {query}"
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def embed_queries_batch(
        self,
        queries: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed multiple queries with 'query: ' prefix.

        Args:
            queries: List of query strings.
            batch_size: Batch size for encoding.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            Normalized embedding matrix of shape (n_queries, dimension).
        """
        texts = [f"query: {q}" for q in queries]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    """Save embeddings to a numpy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    print(f"Saved embeddings ({embeddings.shape}) to {path}")


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings from a numpy file."""
    embeddings = np.load(path)
    print(f"Loaded embeddings ({embeddings.shape}) from {path}")
    return embeddings
