"""Prompt assembly unit smoke."""

from __future__ import annotations

from src.generation.prompts import (
    SYSTEM_PROMPT,
    assemble_context,
    build_user_message,
    hit_supports_answer,
    looks_like_refusal,
    parse_generation,
)
from src.indexing.models import Hit


def test_assemble_context_delimiters() -> None:
    hits = [
        Hit(
            score=0.1,
            document_id="IT-SPEC-2025-A",
            section="Global IT Hardware & Expense Matrix",
            chunk_id="IT-SPEC-2025-A:body:0",
            text="EMEA €1,800",
        )
    ]
    ctx = assemble_context(hits)
    assert "<documents>" in ctx
    assert "</documents>" in ctx
    assert 'id="IT-SPEC-2025-A"' in ctx
    assert 'chunk_id="IT-SPEC-2025-A:body:0"' in ctx
    assert "EMEA €1,800" in ctx


def test_system_prompt_maps_job_titles_to_tiers() -> None:
    lower = SYSTEM_PROMPT.lower()
    assert "tier 1" in lower
    assert "ai software engineer" in lower
    assert "parental leave" in lower


def test_user_message_wraps_question_not_system() -> None:
    msg = build_user_message("laptop Germany?", [])
    assert "<question>" in msg
    assert "laptop Germany?" in msg
    # System rules must not be concatenated into the user channel
    assert "corporate policy assistant" not in msg.lower()
    assert "Answer ONLY using the policy excerpts" not in msg
    assert "Answer ONLY using the policy excerpts" in SYSTEM_PROMPT


def test_looks_like_refusal_only_explicit_flag() -> None:
    assert looks_like_refusal("REFUSAL: not in the documents") is True
    assert (
        looks_like_refusal(
            "Based on the provided context, engineers are fully remote and the allowance is $750."
        )
        is False
    )


def test_parse_generation_strips_trailer_and_flag() -> None:
    answer, used, refused = parse_generation(
        "REFUSAL: The documents do not contain parental leave.\n"
        "USED_CHUNKS:"
    )
    assert refused is True
    assert used == []
    assert "REFUSAL:" not in answer
    assert "parental leave" in answer.lower()


def test_parse_generation_used_chunk_ids() -> None:
    answer, used, refused = parse_generation(
        "Home office is $750.\nUSED_CHUNKS: POLICY-2025-002:2:0, IT-SPEC-2025-A:body:0"
    )
    assert refused is False
    assert "USED_CHUNKS" not in answer
    assert used == ["POLICY-2025-002:2:0", "IT-SPEC-2025-A:body:0"]


def test_hit_supports_answer_money_and_not_unrelated() -> None:
    home = Hit(
        score=0.1,
        document_id="POLICY-2025-002",
        section="2",
        chunk_id="a",
        text="Home office stipend $750 USD per calendar year.",
    )
    table = Hit(
        score=0.1,
        document_id="IT-SPEC-2025-A",
        section="matrix",
        chunk_id="b",
        text="EMEA | Laptop €1,800 EUR | Mobile €60 EUR",
    )
    answer = "The home office allowance is $750 USD per year."
    assert hit_supports_answer(home, answer) is True
    assert hit_supports_answer(table, answer) is False
