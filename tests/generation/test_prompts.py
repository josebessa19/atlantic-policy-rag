"""Prompt assembly unit smoke."""

from __future__ import annotations

from src.generation.prompts import SYSTEM_PROMPT, assemble_context, build_user_message
from src.indexing.models import Hit


def test_assemble_context_delimiters() -> None:
    hits = [
        Hit(
            score=0.1,
            document_id="IT-SPEC-2025-A",
            section="Global IT Hardware & Expense Matrix",
            chunk_id="a",
            text="EMEA €1,800",
        )
    ]
    ctx = assemble_context(hits)
    assert "<documents>" in ctx
    assert "</documents>" in ctx
    assert 'id="IT-SPEC-2025-A"' in ctx
    assert "EMEA €1,800" in ctx


def test_user_message_wraps_question_not_system() -> None:
    msg = build_user_message("laptop Germany?", [])
    assert "<question>" in msg
    assert "laptop Germany?" in msg
    # System rules must not be concatenated into the user channel
    assert "corporate policy assistant" not in msg.lower()
    assert "Answer ONLY using the policy excerpts" not in msg
    assert "Answer ONLY using the policy excerpts" in SYSTEM_PROMPT
