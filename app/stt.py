"""Sarvam AI Speech-to-Text wrapper with retry, backoff, and timeout.

Wraps the Sarvam `saaras:v3` STT API with:
- Retry with exponential backoff (configurable retries)
- Timeout handling
- Fallback on failure
"""

from __future__ import annotations

import io
import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.schemas import STTResponse
from app.config import settings


class STTError(Exception):
    """STT processing error."""
    pass


class STTClient:
    """Sarvam AI Speech-to-Text client with resilience patterns."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy initialization of the Sarvam client."""
        if self._client is None:
            from sarvamai import SarvamAI
            self._client = SarvamAI(api_subscription_key=settings.sarvam_api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        reraise=True,
    )
    def _transcribe_with_retry(self, audio_data: bytes, language_hint: str | None = None) -> dict:
        """Call Sarvam STT with retry logic.

        Args:
            audio_data: Raw audio bytes (WAV format).
            language_hint: Optional language code hint.

        Returns:
            Raw response from Sarvam API.
        """
        client = self._get_client()

        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"

        response = client.speech_to_text.transcribe(
            file=audio_file,
            model=settings.stt_model,
            mode="transcribe",
        )

        return response

    async def transcribe(self, audio_data: bytes, language_hint: str | None = None) -> STTResponse:
        """Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes.
            language_hint: Optional language code hint.

        Returns:
            Typed STTResponse with transcript and metadata.

        Raises:
            STTError: If transcription fails after retries.
        """
        start = time.perf_counter()

        try:
            import asyncio
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._transcribe_with_retry(audio_data, language_hint),
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Parse response
            transcript = ""
            detected_lang = language_hint or "en"
            confidence = 0.0

            if hasattr(response, "transcript"):
                transcript = response.transcript or ""
            elif isinstance(response, dict):
                transcript = response.get("transcript", "")
            elif isinstance(response, str):
                transcript = response
            
            if hasattr(response, "language_code"):
                detected_lang = response.language_code or detected_lang
            elif isinstance(response, dict):
                detected_lang = response.get("language_code", detected_lang)

            # Map Sarvam language codes to our codes
            lang_map = {
                "hi-IN": "hi", "ta-IN": "ta", "en-IN": "en",
                "bn-IN": "bn", "te-IN": "te", "mr-IN": "mr",
                "gu-IN": "gu", "kn-IN": "kn", "ml-IN": "ml",
                "pa-IN": "pa", "or-IN": "or", "as-IN": "as",
                "ur-IN": "ur",
            }
            detected_lang = lang_map.get(detected_lang, detected_lang)

            return STTResponse(
                transcript=transcript,
                detected_language=detected_lang,
                confidence=confidence,
                latency_ms=round(elapsed_ms, 2),
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            raise STTError(
                f"Speech-to-text failed after retries: {str(e)}. "
                f"Please try again or use text input."
            ) from e


# Singleton
stt_client = STTClient()
