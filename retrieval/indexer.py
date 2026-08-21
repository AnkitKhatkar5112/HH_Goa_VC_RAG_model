"""ChromaDB index for persistent vector search.

Supports building, saving, loading, and querying the index.
Also supports metadata filtering.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
import numpy as np

import chromadb
from chromadb.config import Settings as ChromaSettings
import faiss

from retrieval.chunking import Chunk


@dataclass
class SearchResult:
    """A single search result from the index."""
    chunk_id: str
    score: float
    rank: int
    text: str
    language: str
    passage_id: str
    strategy_name: str


class ChromaIndex:
    """ChromaDB manager for multiple chunking strategy collections."""

    def __init__(
        self,
        persist_directory: str,
        dimension: int = 384, # e5-small dimension
    ):
        """Initialize the ChromaDB persistent client.

        Args:
            persist_directory: Where to store the DB on disk.
            dimension: Vector dimension.
        """
        self.persist_directory = persist_directory
        self.dimension = dimension

        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collections = {}

    def build(self, embeddings: np.ndarray, chunks: list[Chunk], collection_name: str) -> None:
        """Build/add to a Chroma collection from embeddings.

        Args:
            embeddings: Embedding matrix of shape (n, dimension).
            chunks: Corresponding chunks.
            collection_name: Name of the collection (usually strategy name).
        """
        assert embeddings.shape[0] == len(chunks), "Embeddings and chunks must have same length"
        
        # We explicitly enforce 'cosine' similarity so scores are comparable across collections
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"Adding {len(chunks)} vectors to Chroma collection '{collection_name}'...")

        # Batch adding to avoid memory/grpc limits
        batch_size = 5461 # Chroma recommended batch size
        
        start = time.perf_counter()
        
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            batch_chunks = chunks[i:end]
            batch_embeddings = embeddings[i:end]

            ids = [c.chunk_id for c in batch_chunks]
            # Convert embedding numpy arrays to lists of floats
            emb_list = [emb.tolist() for emb in batch_embeddings]
            metadatas = []
            documents = []
            
            for c in batch_chunks:
                # Chroma metadata values must be strings, ints, or floats
                meta = {}
                for k, v in c.metadata.items():
                    if isinstance(v, bool):
                        meta[k] = "true" if v else "false"
                    elif v is not None:
                        meta[k] = str(v)
                
                # Add base metadata
                meta["language"] = c.language
                meta["passage_id"] = c.passage_id
                meta["strategy"] = c.strategy
                
                metadatas.append(meta)
                documents.append(c.text)

            collection.upsert(
                ids=ids,
                embeddings=emb_list,
                metadatas=metadatas,
                documents=documents
            )

        build_time = time.perf_counter() - start
        self.collections[collection_name] = collection
        print(f"Collection '{collection_name}' updated in {build_time:.2f}s | Total vectors: {collection.count()}")

    def load(self, collection_name: str) -> None:
        """Load a collection reference."""
        try:
            collection = self.client.get_collection(name=collection_name)
            self.collections[collection_name] = collection
            print(f"Loaded collection '{collection_name}' | Total vectors: {collection.count()}")
        except Exception as e:
            print(f"Could not load collection '{collection_name}': {e}")

    def delete(self, collection_name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            print(f"Deleted collection '{collection_name}'")
        except Exception as e:
            print(f"Collection '{collection_name}' does not exist or could not be deleted: {e}")

    def search(
        self,
        query_embedding: np.ndarray,
        collection_name: str,
        top_k: int = 5,
        language_filter: str | None = None,
        is_selected_filter: bool | None = None,
    ) -> list[SearchResult]:
        """Search a specific collection for nearest neighbors.

        Args:
            query_embedding: Query vector of shape (1, dimension).
            collection_name: Name of the collection to search.
            top_k: Number of results to return.
            language_filter: If set, only return results in this language.
            is_selected_filter: If set, filter by is_selected.

        Returns:
            List of SearchResult, sorted by score (descending).
        """
        if collection_name not in self.collections:
            self.load(collection_name)
            
        collection = self.collections.get(collection_name)
        if not collection:
             raise RuntimeError(f"Collection '{collection_name}' not found.")

        # Build where clause
        where_clause = {}
        if language_filter:
            where_clause["language"] = language_filter
        if is_selected_filter is not None:
            where_clause["is_selected"] = "true" if is_selected_filter else "false"
            
        if not where_clause:
            where_clause = None
        elif len(where_clause) > 1:
            where_clause = {"$and": [{k: {"$eq": v}} for k, v in where_clause.items()]}

        emb_list = query_embedding.tolist()
        
        results = collection.query(
            query_embeddings=emb_list,
            n_results=top_k,
            where=where_clause,
            include=["distances", "documents", "metadatas"]
        )

        search_results = []
        if results and results['ids'] and len(results['ids']) > 0:
            ids = results['ids'][0]
            distances = results['distances'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            for rank, (chunk_id, dist, doc, meta) in enumerate(zip(ids, distances, documents, metadatas)):
                # Chroma's cosine returns distance = 1 - cosine_similarity
                # We want similarity score (higher is better)
                score = 1.0 - dist
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    rank=rank,
                    text=doc,
                    language=meta.get("language", "unknown"),
                    passage_id=meta.get("passage_id", ""),
                    strategy_name=meta.get("strategy", collection_name),
                ))

        return search_results


class FAISSIndex:
    """In-memory FAISS index for rapid evaluation."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Inner product with normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        
    def build(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Build the FAISS index from embeddings and chunks."""
        assert embeddings.shape[0] == len(chunks), "Embeddings and chunks must have same length"
        # Ensure float32 for faiss and make a copy so we can normalize safely
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Search the FAISS index."""
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                score=float(dist),
                rank=rank,
                text=chunk.text,
                language=chunk.language,
                passage_id=chunk.passage_id,
                strategy_name=chunk.strategy,
            ))
        return results
