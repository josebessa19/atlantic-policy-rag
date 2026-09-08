"""Prompt-injection pattern checks on the raw user query."""

from __future__ import annotations

import re

# Fail-closed phrases from the brief / TEST-05. Case-insensitive.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(prior|previous|above)\s+instructions?",
        r"ignore\s+all\s+prior\s+instructions?",
        r"print\s+(the\s+)?system\s+prompt",
        r"system\s+override",
        r"return\s+['\"]?APPROVED['\"]?",
    )
)


def is_injection_attempt(query: str) -> bool:
    """True when the user string looks like an instruction override."""
    if not query or not str(query).strip():
        return False
    text = str(query)
    return any(p.search(text) for p in _INJECTION_PATTERNS)
