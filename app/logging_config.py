"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import structlog
import uuid
import time
from contextlib import asynccontextmanager
from typing import Any


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output and standard library integration."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def generate_trace_id() -> str:
    """Generate a short, unique trace ID for request tracing."""
    return uuid.uuid4().hex[:12]


class StageTimer:
    """Context manager for timing pipeline stages with structured logging."""

    def __init__(self, stage_name: str, logger: Any, **extra: Any):
        self.stage_name = stage_name
        self.logger = logger
        self.extra = extra
        self.start_time: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self) -> "StageTimer":
        self.start_time = time.perf_counter()
        self.logger.info(
            f"stage_start",
            stage=self.stage_name,
            **self.extra,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        log_method = self.logger.error if exc_type else self.logger.info
        log_method(
            f"stage_end",
            stage=self.stage_name,
            elapsed_ms=round(self.elapsed_ms, 2),
            success=exc_type is None,
            **self.extra,
        )
