"""Thin Gemini Flash client for grounded generation only."""

from __future__ import annotations

from src.generation.config import gemini_api_key, llm_model
from src.generation.prompts import SYSTEM_PROMPT, build_user_message
from src.indexing.models import Hit


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
