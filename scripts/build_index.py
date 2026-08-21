"""Standalone script to build the ChromaDB index.

Streams the MSMARCO-XI dataset, chunks it across multiple strategies,
embeds the chunks, and writes them to the local persistent ChromaDB.
Run this once before starting the application.
"""

import sys
import os

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import settings
from retrieval.chunking import get_chunker
from retrieval.embedder import Embedder
from retrieval.indexer import ChromaIndex


def main():
    print(f"\n{'='*60}")
    print("Building VoiceRAG ChromaDB Index (Row-batched)")
    print(f"{'='*60}\n")

    # Initialize embedder and persistent DB
    embedder = Embedder(model_name=settings.embedding_model)
    index = ChromaIndex(
        persist_directory=settings.chroma_db_dir,
        dimension=embedder.dimension
    )

    strategies = ["fixed_size", "semantic", "metadata_aware"]
    sample_size = settings.dataset_sample_size
    checkpoint_file = f"data/checkpoint_global_{sample_size}.txt"

    start_row = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            content = f.read().strip()
            if content:
                start_row = int(content)
                print(f"Resuming from row index {start_row}")
    else:
        # Wipe collections if no checkpoint
        for strategy in strategies:
            if strategy in index.collections:
                print(f"Collection {strategy} vector count BEFORE: {index.collections[strategy].count()}")
            index.delete(strategy)

    # Make sure collections are loaded
    for strategy in strategies:
        index.load(strategy)

    chunkers = {s: get_chunker(s) for s in strategies}

    import time
    from retrieval.data_loader import load_msmarco_xi_stream_batches, save_dataset_split

    batch_size = 1000  # Pull 1000 rows at a time
    
    stream = load_msmarco_xi_stream_batches(
        languages=settings.languages_list,
        sample_size=sample_size,
        batch_size=batch_size,
        start_offset=start_row,
    )

    for batch_split in stream:
        batch_start = time.time()
        
        # Save evaluation pairs
        if batch_split.eval_pairs:
            # We need to append them to the existing file
            import json
            from dataclasses import asdict
            os.makedirs("data/processed", exist_ok=True)
            with open("data/processed/eval_pairs.jsonl", "a", encoding="utf-8") as f:
                for ep in batch_split.eval_pairs:
                    f.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")
        
        for strategy in strategies:
            chunker = chunkers[strategy]
            chunks, _ = chunker.chunk(batch_split.passages)
            print(f"[{strategy}] Embedding and upserting {len(chunks)} chunks from this row-batch...")
            if chunks:
                embeddings = embedder.embed_passages(chunks)
                index.build(embeddings, chunks, collection_name=strategy)
                
        # Approximate rows processed by eval_pairs count or batch_size
        # Since eval_pairs map 1:1 to examples generally
        rows_processed = batch_size
        start_row += rows_processed
        
        with open(checkpoint_file, "w") as f:
            f.write(str(start_row))
            
        batch_elapsed = time.time() - batch_start
        print(f"Row-batch completed in {batch_elapsed:.2f}s ({rows_processed/batch_elapsed*60:.2f} rows/min)\n")

    for strategy in strategies:
        print(f"Collection {strategy} vector count AFTER: {index.collections[strategy].count()}")

    print(f"\n{'='*60}")
    print(f"Index build complete. Data persisted to {settings.chroma_db_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
