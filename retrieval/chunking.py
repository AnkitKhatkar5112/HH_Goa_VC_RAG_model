"""Five chunking strategies for MSMARCO-XI passages.

Each chunker implements a common interface: takes a list of Passages,
returns a list of Chunks with metadata.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import nltk

# Download punkt_tab for sentence tokenization (silent if already present)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from retrieval.data_loader import Passage


@dataclass
class Chunk:
    """A text chunk ready for embedding and indexing."""
    chunk_id: str
    text: str
    language: str
    passage_id: str  # parent passage
    strategy: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ParentMapping:
    """Maps child chunk IDs to parent passage texts for parent-document retrieval."""
    mapping: dict[str, str] = field(default_factory=dict)  # chunk_id -> parent_text


class Chunker(ABC):
    """Abstract base class for chunking strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def chunk(self, passages: list[Passage]) -> tuple[list[Chunk], ParentMapping | None]:
        """Chunk passages into smaller units.

        Returns:
            Tuple of (chunks, optional parent_mapping for parent-doc strategy).
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Fixed-size with overlap
# ─────────────────────────────────────────────────────────────────────────────

class FixedSizeChunker(Chunker):
    """Split passages into fixed-size token chunks with overlap.

    This is the baseline control strategy.
    """

    strategy_name = "fixed_size"

    def __init__(self, chunk_size: int = 256, overlap_ratio: float = 0.2):
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)

    def chunk(self, passages: list[Passage]) -> tuple[list[Chunk], None]:
        chunks = []
        for passage in passages:
            words = passage.text.split()
            if len(words) <= self.chunk_size:
                # Passage fits in one chunk
                chunks.append(Chunk(
                    chunk_id=f"{passage.passage_id}_fc0",
                    text=passage.text,
                    language=passage.language,
                    passage_id=passage.passage_id,
                    strategy=self.strategy_name,
                    metadata={"source_lang": passage.source_lang, "target_lang": passage.target_lang},
                ))
            else:
                # Split into overlapping chunks
                start = 0
                ci = 0
                while start < len(words):
                    end = min(start + self.chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    chunks.append(Chunk(
                        chunk_id=f"{passage.passage_id}_fc{ci}",
                        text=chunk_text,
                        language=passage.language,
                        passage_id=passage.passage_id,
                        strategy=self.strategy_name,
                        metadata={"source_lang": passage.source_lang, "target_lang": passage.target_lang},
                    ))
                    ci += 1
                    start = end - self.overlap
                    if end == len(words):
                        break

        return chunks, None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Semantic / Sentence-boundary splitting
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker(Chunker):
    """Split on sentence boundaries, grouping sentences into semantic units.

    Uses NLTK sentence tokenizer for multilingual support.
    Groups consecutive sentences up to max_chunk_words.
    """

    strategy_name = "semantic"

    def __init__(self, max_chunk_words: int = 256, min_chunk_words: int = 30):
        self.max_chunk_words = max_chunk_words
        self.min_chunk_words = min_chunk_words

    def _sent_tokenize(self, text: str, language: str) -> list[str]:
        """Sentence tokenization with language-aware fallback."""
        lang_map = {"en": "english", "hi": "english", "ta": "english", "bn": "english"}
        lang_name = lang_map.get(language, "english")
        try:
            sentences = nltk.sent_tokenize(text, language=lang_name)
        except Exception:
            # Fallback: split on common sentence-ending punctuation
            sentences = re.split(r'(?<=[.!?।॥])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, passages: list[Passage]) -> tuple[list[Chunk], None]:
        chunks = []
        for passage in passages:
            sentences = self._sent_tokenize(passage.text, passage.language)

            if not sentences:
                continue

            current_chunk_sents: list[str] = []
            current_word_count = 0
            ci = 0

            for sent in sentences:
                sent_words = len(sent.split())
                if current_word_count + sent_words > self.max_chunk_words and current_chunk_sents:
                    # Flush current chunk
                    chunk_text = " ".join(current_chunk_sents)
                    chunks.append(Chunk(
                        chunk_id=f"{passage.passage_id}_sc{ci}",
                        text=chunk_text,
                        language=passage.language,
                        passage_id=passage.passage_id,
                        strategy=self.strategy_name,
                        metadata={"source_lang": passage.source_lang, "target_lang": passage.target_lang},
                    ))
                    ci += 1
                    current_chunk_sents = []
                    current_word_count = 0

                current_chunk_sents.append(sent)
                current_word_count += sent_words

            # Flush remaining
            if current_chunk_sents:
                chunk_text = " ".join(current_chunk_sents)
                # Merge with previous if too small
                if len(chunk_text.split()) < self.min_chunk_words and chunks and chunks[-1].passage_id == passage.passage_id:
                    chunks[-1].text += " " + chunk_text
                else:
                    chunks.append(Chunk(
                        chunk_id=f"{passage.passage_id}_sc{ci}",
                        text=chunk_text,
                        language=passage.language,
                        passage_id=passage.passage_id,
                        strategy=self.strategy_name,
                        metadata={"source_lang": passage.source_lang, "target_lang": passage.target_lang},
                    ))

        return chunks, None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Parent-document retrieval (small-to-big)
# ─────────────────────────────────────────────────────────────────────────────

class ParentDocumentChunker(Chunker):
    """Index small sub-chunks for precise matching; return parent passage for generation.

    Creates small chunks (128 words) for retrieval, but maintains a mapping
    back to the full parent passage text for the generation context.
    """

    strategy_name = "parent_document"

    def __init__(self, child_chunk_size: int = 128, overlap_ratio: float = 0.15):
        self.child_chunk_size = child_chunk_size
        self.overlap = int(child_chunk_size * overlap_ratio)

    def chunk(self, passages: list[Passage]) -> tuple[list[Chunk], ParentMapping]:
        chunks = []
        parent_mapping = ParentMapping()

        for passage in passages:
            words = passage.text.split()

            if len(words) <= self.child_chunk_size:
                # Small enough to be its own chunk; parent = itself
                cid = f"{passage.passage_id}_pd0"
                chunks.append(Chunk(
                    chunk_id=cid,
                    text=passage.text,
                    language=passage.language,
                    passage_id=passage.passage_id,
                    strategy=self.strategy_name,
                    metadata={"source_lang": passage.source_lang, "target_lang": passage.target_lang},
                ))
                parent_mapping.mapping[cid] = passage.text
            else:
                start = 0
                ci = 0
                while start < len(words):
                    end = min(start + self.child_chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    cid = f"{passage.passage_id}_pd{ci}"
                    chunks.append(Chunk(
                        chunk_id=cid,
                        text=chunk_text,
                        language=passage.language,
                        passage_id=passage.passage_id,
                        strategy=self.strategy_name,
                        metadata={"source_lang": passage.source_lang, "target_lang": passage.target_lang},
                    ))
                    # Map back to full parent passage
                    parent_mapping.mapping[cid] = passage.text
                    ci += 1
                    start = end - self.overlap
                    if end == len(words):
                        break

        return chunks, parent_mapping


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: Metadata-aware chunking
# ─────────────────────────────────────────────────────────────────────────────

class MetadataAwareChunker(Chunker):
    """Combines query text and passage text into a single tightly-scoped chunk.

    Tags chunks with query_type, is_selected, and language metadata, enabling
    metadata-filtered retrieval at query time.
    """

    strategy_name = "metadata_aware"

    def __init__(self, chunk_size: int = 256, overlap_ratio: float = 0.2):
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)

    def chunk(self, passages: list[Passage]) -> tuple[list[Chunk], None]:
        chunks = []
        for passage in passages:
            if not passage.is_selected:
                continue
            
            # Bundle query text and passage text
            combined_text = f"Query: {passage.query_text}\nPassage: {passage.text}"
            words = combined_text.split()
            
            metadata = {
                "source_lang": passage.source_lang,
                "target_lang": passage.target_lang,
                "language": passage.language,
                "is_selected": passage.is_selected,
                "query_id": passage.query_id,
            }

            if len(words) <= self.chunk_size:
                chunks.append(Chunk(
                    chunk_id=f"{passage.passage_id}_ma0",
                    text=combined_text,
                    language=passage.language,
                    passage_id=passage.passage_id,
                    strategy=self.strategy_name,
                    metadata=metadata,
                ))
            else:
                start = 0
                ci = 0
                while start < len(words):
                    end = min(start + self.chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    chunks.append(Chunk(
                        chunk_id=f"{passage.passage_id}_ma{ci}",
                        text=chunk_text,
                        language=passage.language,
                        passage_id=passage.passage_id,
                        strategy=self.strategy_name,
                        metadata=metadata,
                    ))
                    ci += 1
                    start = end - self.overlap
                    if end == len(words):
                        break

        return chunks, None


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

CHUNKER_REGISTRY: dict[str, type[Chunker]] = {
    "fixed_size": FixedSizeChunker,
    "semantic": SemanticChunker,
    "parent_document": ParentDocumentChunker,
    "metadata_aware": MetadataAwareChunker,
}


def get_chunker(strategy: str, **kwargs) -> Chunker:
    """Get a chunker by strategy name."""
    if strategy not in CHUNKER_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(CHUNKER_REGISTRY.keys())}")
    return CHUNKER_REGISTRY[strategy](**kwargs)
