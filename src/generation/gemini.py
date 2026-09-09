"""Thin Gemini Flash client for grounded generation only."""

from __future__ import annotations

import logging
import time

from src.generation.config import gemini_api_key, llm_model
from src.generation.prompts import SYSTEM_PROMPT, build_user_message
from src.indexing.models import Hit

logger = logging.getLogger(__name__)

_TRANSIENT_MARKERS = ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")
_MAX_ATTEMPTS = 3


def _is_transient_gemini_error(exc: BaseException) -> bool:
    text = str(exc)
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (429, 503):
        return True
    return any(token in text for token in _TRANSIENT_MARKERS)


def generate_answer(query: str, hits: list[Hit]) -> str:
    """Call Gemini with system_instruction separate from user/document data."""
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key "
            "(generation only — embeddings do not use Gemini)."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    user_message = build_user_message(query, hits)
    last_exc: BaseException | None = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=llm_model(),
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                ),
            )
            text = getattr(response, "text", None)
            if not text or not str(text).strip():
                raise RuntimeError("Gemini returned an empty response")
            return str(text).strip()
        except Exception as exc:  # noqa: BLE001 — retry quota / blips only
            last_exc = exc
            if attempt >= _MAX_ATTEMPTS - 1 or not _is_transient_gemini_error(exc):
                raise
            wait = 2 * (attempt + 1)
            logger.warning(
                "Transient Gemini error (attempt %s/%s), retry in %ss: %s",
                attempt + 1,
                _MAX_ATTEMPTS,
                wait,
                exc,
            )
            time.sleep(wait)

    raise last_exc or RuntimeError("Gemini request failed")
