"""Cheap token estimation for chunk size caps (no hard tiktoken dependency)."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Approximate token count as ~4 characters per token (ceil)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def format_path_prefix(section_path: str) -> str:
    if not section_path:
        return ""
    return f"[{section_path}]"


def embed_text(section_path: str, body: str) -> str:
    """Text used for embedding: optional path prefix + body."""
    body = body.strip()
    prefix = format_path_prefix(section_path)
    if prefix:
        return f"{prefix}\n{body}" if body else prefix
    return body
