"""Prompt-injection pattern checks on the raw user query and model output."""

from __future__ import annotations

import re

# Fail-closed intent classes (override / exfiltrate prompt / unauthorized approval).
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(prior|previous|above)\s+instructions?",
        r"(forget|disregard)\s+(all\s+)?((your|the)\s+)?(prior|previous|above)?\s*instructions?",
        r"(print|reveal|output|show|dump)\s+(the\s+)?system\s+prompt",
        r"system\s+override",
        r"(return|output|print)\s+['\"]?APPROVED['\"]?",
    )
)

# Distinctive phrases from SYSTEM_PROMPT — must never appear in a user-facing answer.
_LEAK_MARKERS = (
    "corporate policy assistant",
    "answer only using the policy excerpts",
    "inside <documents>",
)

# Uppercase token only — do not match "approval" / "not approved" in policy text.
_APPROVED_TOKEN = re.compile(r"\bAPPROVED\b")


def is_injection_attempt(query: str) -> bool:
    """True when the user string looks like an instruction override."""
    if not query or not str(query).strip():
        return False
    text = str(query)
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def is_injection_output(answer: str) -> bool:
    """True when the model leaked the system prompt or granted APPROVED."""
    if not answer or not str(answer).strip():
        return False
    text = str(answer)
    lower = text.lower()
    if any(marker in lower for marker in _LEAK_MARKERS):
        return True
    return _APPROVED_TOKEN.search(text) is not None
