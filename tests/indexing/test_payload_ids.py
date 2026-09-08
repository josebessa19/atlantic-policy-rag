"""Payload mapping and UUID5 point-id stability."""

from __future__ import annotations

from datetime import date

from src.ingestion.chunking.models import Chunk
from src.indexing.ids import point_id
from src.indexing.store import chunk_to_payload


def _chunk(**kwargs) -> Chunk:
    base = dict(
        id="IT-SPEC-2025-A:body:0",
        text="Document IT-SPEC-2025-A (Active) | Section: Global IT\n| EMEA | €1,800 |",
        section_path="",
        source="IT_Hardware_Allowance_Matrix.txt",
        chunk_type="table",
        token_estimate=20,
        document_id="IT-SPEC-2025-A",
        status="active",
        effective_date=None,
        supersedes=[],
        title="Global IT Hardware & Expense Matrix",
    )
    base.update(kwargs)
    return Chunk(**base)


def test_point_id_stable() -> None:
    a = point_id("POLICY-2025-002:2-home-office-allowance-update:0")
    b = point_id("POLICY-2025-002:2-home-office-allowance-update:0")
    assert a == b
    assert a.version == 5


def test_point_id_differs_by_chunk() -> None:
    assert point_id("doc:a:0") != point_id("doc:a:1")


def test_payload_minimum_fields() -> None:
    c = _chunk(
        section_path="2. Home Office Allowance Update",
        effective_date=date(2025, 1, 1),
        supersedes=["POLICY-2024-001"],
        title="Enterprise Remote Work Policy (2025 Revised)",
        document_id="POLICY-2025-002",
        id="POLICY-2025-002:2-home-office-allowance-update:0",
        chunk_type="section",
    )
    p = chunk_to_payload(c)
    for key in (
        "document_id",
        "source",
        "section",
        "chunk_type",
        "text",
        "status",
        "effective_date",
        "supersedes",
        "title",
        "chunk_id",
    ):
        assert key in p
    assert p["document_id"] == "POLICY-2025-002"
    assert p["section"] == "2. Home Office Allowance Update"
    assert p["effective_date"] == "2025-01-01"
    assert p["supersedes"] == ["POLICY-2024-001"]
    assert p["chunk_id"] == c.id


def test_section_falls_back_to_title() -> None:
    p = chunk_to_payload(_chunk(section_path=""))
    assert p["section"] == "Global IT Hardware & Expense Matrix"
