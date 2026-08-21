import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever import RetrieverService
from retrieval.data_loader import _stream_examples, _process_example

def run_diagnostics():
    print("="*60)
    print("HYPOTHESIS 1 & 2 DIAGNOSTICS")
    print("="*60)
    
    retriever = RetrieverService()
    retriever.load()
    
    print("Streaming examples to find an actual eval query...")
    examples = _stream_examples("hi", 500)
    
    eval_pairs = []
    indexed_passage_ids = set()
    
    for idx, ex in enumerate(examples):
        passages, ep, en_passages, en_ep = _process_example(ex, "hi", idx, extract_english=True)
        if ep: eval_pairs.append(ep)
        if en_ep: eval_pairs.append(en_ep)
        for p in passages + en_passages:
            indexed_passage_ids.add(p.passage_id)
            
    print(f"\nCoverage check on {len(eval_pairs)} eval pairs:")
    missing_count = sum(1 for ep in eval_pairs if any(pid not in indexed_passage_ids for pid in ep.relevant_passage_ids))
    print(f"- Eval pairs with ALL relevant passages in index: {len(eval_pairs) - missing_count}")
    print(f"- Eval pairs with MISSING relevant passages: {missing_count}")
    
    if not eval_pairs:
        return
        
    # Use the first Hindi eval pair
    ep = next(e for e in eval_pairs if e.language == "hi")
    query = ep.query
    print(f"\nTest Query (from eval set): '{query}'")
    print(f"Target Relevant Passage ID: {ep.relevant_passage_ids}")
    
    query_emb = retriever.embedder.embed_query(query)
    
    for strategy in retriever.strategies:
        print(f"\n--- Strategy: {strategy} ---")
        results = retriever.index.search(
            query_embedding=query_emb,
            collection_name=strategy,
            top_k=1,
            language_filter=None,
            is_selected_filter=None
        )
        if results:
            res = results[0]
            print(f"Top Score: {res.score:.4f}")
            print(f"Chunk ID: {res.chunk_id}")
            print(f"Match? {'YES' if any(pid in res.chunk_id for pid in ep.relevant_passage_ids) else 'NO'}")
            print(f"Chunk Text Preview: {res.text[:150]}...")
        else:
            print("No results found.")

if __name__ == "__main__":
    run_diagnostics()
