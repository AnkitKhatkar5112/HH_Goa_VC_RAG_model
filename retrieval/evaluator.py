"""Retrieval quality evaluator.

Compares chunking/retrieval strategies head-to-head using Recall@k and MRR
on the held-out query/answer evaluation set from MSMARCO-XI.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
from tabulate import tabulate

from retrieval.chunking import Chunk, Chunker, ParentMapping, get_chunker, CHUNKER_REGISTRY
from retrieval.data_loader import DatasetSplit, Passage, EvalPair
from retrieval.embedder import Embedder
from retrieval.indexer import FAISSIndex


@dataclass
class StrategyResult:
    """Evaluation results for a single chunking strategy."""
    strategy: str
    num_chunks: int
    recall_at_5: float
    mrr: float
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float


def evaluate_strategy(
    strategy_name: str,
    passages: list[Passage],
    eval_pairs: list[EvalPair],
    embedder: Embedder,
    max_eval_queries: int = 500,
    top_k: int = 5,
) -> StrategyResult:
    """Evaluate a single chunking strategy.

    Args:
        strategy_name: Name of the chunking strategy.
        passages: List of all passages.
        eval_pairs: Evaluation query-answer pairs.
        embedder: The embedding model.
        max_eval_queries: Maximum number of eval queries to use.
        top_k: Number of results to retrieve.

    Returns:
        StrategyResult with metrics.
    """
    print(f"\n{'─'*60}")
    print(f"Evaluating strategy: {strategy_name}")
    print(f"{'─'*60}")

    # Step 1: Chunk
    chunker = get_chunker(strategy_name)
    chunks, parent_mapping = chunker.chunk(passages)
    print(f"  Chunks created: {len(chunks)}")

    # Build a mapping: passage_id -> [chunk_ids]
    pid_to_cids: dict[str, list[str]] = {}
    for c in chunks:
        pid_to_cids.setdefault(c.passage_id, []).append(c.chunk_id)

    # Step 2: Embed chunks
    print(f"  Embedding {len(chunks)} chunks...")
    chunk_embeddings = embedder.embed_passages(chunks, batch_size=128)

    # Step 3: Build FAISS index
    faiss_index = FAISSIndex(dimension=embedder.dimension)
    faiss_index.build(chunk_embeddings, chunks)

    # Step 4: Evaluate
    # Filter eval pairs to only those with known relevant passage IDs
    valid_pairs = [ep for ep in eval_pairs if ep.relevant_passage_ids]
    if len(valid_pairs) > max_eval_queries:
        # Stratified sample
        np.random.seed(42)
        indices = np.random.choice(len(valid_pairs), max_eval_queries, replace=False)
        valid_pairs = [valid_pairs[i] for i in indices]

    print(f"  Evaluating on {len(valid_pairs)} queries (top_k={top_k})...")

    reciprocal_ranks = []
    recall_hits = 0
    latencies = []

    for ep in valid_pairs:
        # Get all chunk IDs for the relevant passages
        relevant_chunk_ids = set()
        for pid in ep.relevant_passage_ids:
            if pid in pid_to_cids:
                relevant_chunk_ids.update(pid_to_cids[pid])

        if not relevant_chunk_ids:
            continue

        # Query
        query_emb = embedder.embed_query(ep.query)

        start = time.perf_counter()
        results = faiss_index.search(query_emb, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        retrieved_ids = [r.chunk_id for r in results]

        # Recall@k: did any relevant chunk appear in top-k?
        if relevant_chunk_ids & set(retrieved_ids):
            recall_hits += 1

        # MRR: reciprocal rank of first relevant result
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids):
            if rid in relevant_chunk_ids:
                rr = 1.0 / (rank + 1)
                break
        reciprocal_ranks.append(rr)

    total_queries = len(reciprocal_ranks)
    if total_queries == 0:
        print("  ⚠ No valid evaluation queries with known relevant passages.")
        return StrategyResult(
            strategy=strategy_name,
            num_chunks=len(chunks),
            recall_at_5=0.0,
            mrr=0.0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p90_latency_ms=0.0,
        )

    recall_at_k = recall_hits / total_queries
    mrr = sum(reciprocal_ranks) / total_queries
    latencies_arr = np.array(latencies)

    result = StrategyResult(
        strategy=strategy_name,
        num_chunks=len(chunks),
        recall_at_5=round(recall_at_k, 4),
        mrr=round(mrr, 4),
        avg_latency_ms=round(float(np.mean(latencies_arr)), 2),
        p50_latency_ms=round(float(np.percentile(latencies_arr, 50)), 2),
        p90_latency_ms=round(float(np.percentile(latencies_arr, 90)), 2),
    )

    print(f"  Results: Recall@{top_k}={result.recall_at_5:.4f}, MRR={result.mrr:.4f}, "
          f"Avg latency={result.avg_latency_ms:.2f}ms, P50={result.p50_latency_ms:.2f}ms")

    return result


def run_full_evaluation(
    dataset: DatasetSplit,
    embedder: Embedder,
    strategies: list[str] | None = None,
    output_dir: str = "bench/results",
    max_eval_queries: int = 500,
) -> list[StrategyResult]:
    """Run head-to-head comparison of all chunking strategies.

    Args:
        dataset: The loaded dataset split.
        embedder: The embedding model.
        strategies: List of strategy names to evaluate (None = all).
        output_dir: Directory for output files.
        max_eval_queries: Max queries per strategy.

    Returns:
        List of StrategyResult objects.
    """
    if strategies is None:
        strategies = list(CHUNKER_REGISTRY.keys())

    print(f"\n{'='*60}")
    print(f"Running chunking strategy evaluation")
    print(f"Strategies: {strategies}")
    print(f"Passages: {len(dataset.passages)}, Eval pairs: {len(dataset.eval_pairs)}")
    print(f"{'='*60}")

    results = []
    for strategy in strategies:
        try:
            result = evaluate_strategy(
                strategy_name=strategy,
                passages=dataset.passages,
                eval_pairs=dataset.eval_pairs,
                embedder=embedder,
                max_eval_queries=max_eval_queries,
            )
            results.append(result)
        except Exception as e:
            print(f"  ✗ Strategy '{strategy}' failed: {e}")

    # Generate comparison table
    if results:
        _save_comparison(results, output_dir)

    return results


def _save_comparison(results: list[StrategyResult], output_dir: str) -> None:
    """Save the comparison table as CSV and formatted markdown."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, "chunking_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Strategy,Num Chunks,Recall@5,MRR,Avg Latency (ms),P50 Latency (ms),P90 Latency (ms)\n")
        for r in results:
            f.write(f"{r.strategy},{r.num_chunks},{r.recall_at_5},{r.mrr},{r.avg_latency_ms},{r.p50_latency_ms},{r.p90_latency_ms}\n")
    print(f"\nSaved comparison CSV to {csv_path}")

    # Markdown table
    table_data = []
    for r in results:
        table_data.append([
            r.strategy,
            r.num_chunks,
            f"{r.recall_at_5:.4f}",
            f"{r.mrr:.4f}",
            f"{r.avg_latency_ms:.2f}",
            f"{r.p50_latency_ms:.2f}",
            f"{r.p90_latency_ms:.2f}",
        ])

    headers = ["Strategy", "Chunks", "Recall@5", "MRR", "Avg Latency (ms)", "P50 (ms)", "P90 (ms)"]
    table_str = tabulate(table_data, headers=headers, tablefmt="github")

    md_path = os.path.join(output_dir, "chunking_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Chunking Strategy Comparison\n\n")
        f.write(table_str)
        f.write("\n")
    print(f"Saved comparison markdown to {md_path}")

    # Print to console
    print(f"\n{table_str}\n")

    # JSON for programmatic use
    json_path = os.path.join(output_dir, "chunking_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{
            "strategy": r.strategy,
            "num_chunks": r.num_chunks,
            "recall_at_5": r.recall_at_5,
            "mrr": r.mrr,
            "avg_latency_ms": r.avg_latency_ms,
            "p50_latency_ms": r.p50_latency_ms,
            "p90_latency_ms": r.p90_latency_ms,
        } for r in results], f, indent=2)
