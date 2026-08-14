"""Application configuration via Pydantic Settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    # ── API Keys ──────────────────────────────────────────────────────────
    sarvam_api_key: str = Field(default="", description="Sarvam AI API key for STT")
    google_api_key: str = Field(default="", description="Google Gemini API key")

    # ── Application ───────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "production"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Embedding & Retrieval ─────────────────────────────────────────────
    embedding_model: str = "intfloat/multilingual-e5-base"
    faiss_index_path: str = "data/faiss_index"
    chunk_strategy: str = "metadata_aware"
    top_k: int = 5

    # ── FAISS HNSW parameters ─────────────────────────────────────────────
    hnsw_m: int = 32
    hnsw_ef_construction: int = 128
    hnsw_ef_search: int = 64

    # ── Guardrail Thresholds ──────────────────────────────────────────────
    off_topic_threshold: float = 0.35
    safety_enabled: bool = True
    grounding_threshold: float = 0.5
    grounding_contradiction_threshold: float = 0.7

    # ── STT Settings ──────────────────────────────────────────────────────
    stt_model: str = "saaras:v3"
    stt_timeout_seconds: float = 10.0
    stt_max_retries: int = 3

    # ── LLM Generation Settings ───────────────────────────────────────────
    llm_model: str = "gemini-2.0-flash"
    llm_timeout_seconds: float = 15.0
    llm_max_retries: int = 3

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_languages: str = "en,hi,ta"
    dataset_sample_size: int = 10000

    # ── NLI Model for Grounding ───────────────────────────────────────────
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # ── Reranker (for hybrid strategy) ────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def languages_list(self) -> list[str]:
        return [lang.strip() for lang in self.dataset_languages.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton instance
settings = Settings()
