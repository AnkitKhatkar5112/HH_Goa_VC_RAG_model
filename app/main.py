"""FastAPI application — main entry point.

Endpoints:
- POST /query       — voice + text query (multipart)
- POST /query/text  — text-only query (JSON)
- GET  /health      — health check
- GET  /metrics     — recent latency stats
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.logging_config import setup_logging
from app.schemas import PipelineResponse, TextQueryRequest, HealthResponse
from app.pipeline import run_pipeline
from app.retriever import retriever_service


# ── Startup / Shutdown ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # Startup
    setup_logging(settings.log_level)
    print(f"\n{'='*60}")
    print(f"Starting Voice-Enabled RAG System")
    print(f"Environment: {settings.app_env}")
    print(f"Languages: {settings.languages_list}")
    print(f"{'='*60}\n")

    # Load retriever (embedding model + FAISS index)
    try:
        retriever_service.load()
    except Exception as e:
        print(f"⚠ Failed to load retriever: {e}")
        print("  Server will start but retrieval will not work.")

    yield  # Server is running

    # Shutdown
    print("Shutting down...")


# ── App Setup ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice-Enabled Multilingual RAG",
    description="Multilingual voice-enabled RAG system on MSMARCO-XI with Sarvam STT",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        index_loaded=retriever_service.is_loaded,
        languages=settings.languages_list,
        total_vectors=retriever_service.index.index.ntotal if retriever_service.is_loaded else 0,
    )


@app.post("/query", response_model=PipelineResponse)
async def query_voice(
    audio: UploadFile | None = File(None),
    text: str | None = Form(None),
    language: str | None = Form(None),
):
    """Voice + text query endpoint (multipart form).

    Accepts either:
    - An audio file (WAV) for STT + RAG
    - A text query for direct RAG
    - Both (text takes priority)
    """
    audio_data = None
    if audio:
        audio_data = await audio.read()

    if not audio_data and not text:
        raise HTTPException(status_code=400, detail="Provide either audio or text query")

    result = await run_pipeline(
        audio_data=audio_data,
        text_query=text,
        language_hint=language,
    )
    return result


@app.post("/query/text", response_model=PipelineResponse)
async def query_text(request: TextQueryRequest):
    """Text-only query endpoint (JSON body).

    Used for benchmarking and text-based queries.
    """
    result = await run_pipeline(
        text_query=request.query,
        language_hint=request.language,
        top_k=request.top_k,
        strategy=request.strategy,
    )
    return result


# ── Static files (Frontend) ─────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
    )
