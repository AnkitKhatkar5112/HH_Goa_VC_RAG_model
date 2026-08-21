import time
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import settings
from retrieval.data_loader import load_msmarco_xi
from retrieval.chunking import get_chunker
from retrieval.embedder import Embedder

def run_test(device, chunks, total_passages):
    print(f"\nTesting on device: {device}")
    embedder = Embedder(model_name=settings.embedding_model, device=device)
    
    # Warmup
    _ = embedder.embed_passages(chunks[:10], show_progress=False)
    
    start_time = time.time()
    embeddings = embedder.embed_passages(chunks, show_progress=False)
    end_time = time.time()
    
    elapsed = end_time - start_time
    chunks_per_min = (len(chunks) / elapsed) * 60
    rows_per_min = (total_passages / elapsed) * 60
    
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Throughput: {chunks_per_min:.2f} chunks/minute ({rows_per_min:.2f} rows/minute)")

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    dataset = load_msmarco_xi(
        languages=["hi"],
        sample_size=500,
    )
    
    chunker = get_chunker("metadata_aware")
    chunks, pm = chunker.chunk(dataset.passages)
    
    print(f"Created {len(chunks)} chunks from {len(dataset.passages)} passages.")
    
    # run_test("cpu", chunks, len(dataset.passages))
    
    if torch.cuda.is_available():
        run_test("cuda", chunks, len(dataset.passages))

if __name__ == "__main__":
    main()
