"""Scratch script to test retrieval directly from ChromaDB."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.retriever import retriever_service

async def main():
    try:
        retriever_service.load()
    except RuntimeError as e:
        print(e)
        return

    test_queries = [
        "what is the capital of india?",
        "भारत की राजधानी क्या है?",
    ]

    for q in test_queries:
        print(f"\n{'='*40}")
        print(f"Query: {q}")
        print(f"{'='*40}")
        
        # Test across all strategies
        response = await retriever_service.retrieve(
            query=q,
            top_k=3,
            use_language_filter=False
        )

        for i, chunk in enumerate(response.chunks):
            print(f"[{i+1}] Score: {chunk.score:.4f} | Lang: {chunk.language}")
            print(f"Text: {chunk.text[:200]}...")
            print("-" * 20)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
