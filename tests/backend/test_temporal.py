"""Assembler must not leak superseded 2024 text after Option A drop."""

from __future__ import annotations

from src.generation.prompts import assemble_context
from src.guardrails.temporal import drop_superseded
from src.indexing.models import Hit


def test_assembler_excludes_2024_after_drop() -> None:
    hits = [
        Hit(
            score=0.9,
            document_id="POLICY-2024-001",
            section="1. Workplace Flexibility",
            chunk_id="POLICY-2024-001:1:0",
            status="legacy",
            text=(
                "Document POLICY-2024-001 (Legacy) | Section: 1. Workplace Flexibility\n"
                "Employees may work remotely for up to two (2) days per week."
            ),
        ),
        Hit(
            score=0.85,
            document_id="POLICY-2025-002",
            section="1. Hybrid & Remote Work Framework",
            chunk_id="POLICY-2025-002:1:0",
            status="active",
            text=(
                "Document POLICY-2025-002 (Active) | Section: 1. Hybrid & Remote Work Framework\n"
                "Tier 1 Roles (Engineering): Fully remote allowed."
            ),
            payload={"supersedes": ["POLICY-2024-001"]},
        ),
    ]
    kept = drop_superseded(hits, {"POLICY-2024-001"})
    context = assemble_context(kept)
    assert "POLICY-2024-001" not in context
    assert "two (2) days" not in context
    assert "POLICY-2025-002" in context
    assert "Fully remote" in context
    assert '<document id="POLICY-2025-002"' in context
