"""Every chunk carries lifecycle metadata; pipeline is deterministic."""

from pathlib import Path

from src.ingestion.pipeline import parse_dir
from src.ingestion.chunking.tokens import dense_index_text

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def test_parse_dir_metadata_on_every_chunk() -> None:
    chunks = parse_dir(RAW_DIR)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.document_id, f"missing document_id on {c.id}"
        assert c.status in ("active", "legacy")
        assert c.source.endswith(".txt")
        assert c.text.startswith("Document "), c.text[:50]
        assert c.id.startswith(f"{c.document_id}:")
        body = dense_index_text(c.text)
        assert body
        assert not body.startswith("Document ")


def test_parse_dir_deterministic() -> None:
    a = parse_dir(RAW_DIR)
    b = parse_dir(RAW_DIR)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_lifecycle_by_document() -> None:
    chunks = parse_dir(RAW_DIR)
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.document_id, []).append(c)

    assert "POLICY-2024-001" in by_doc
    assert all(c.status == "legacy" for c in by_doc["POLICY-2024-001"])

    assert "POLICY-2025-002" in by_doc
    for c in by_doc["POLICY-2025-002"]:
        assert c.status == "active"
        assert c.supersedes == ["POLICY-2024-001"]

    assert "IT-SPEC-2025-A" in by_doc
    assert all(c.status == "active" for c in by_doc["IT-SPEC-2025-A"])


def test_stable_chunk_ids_format() -> None:
    chunks = parse_dir(RAW_DIR)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    allowance = [
        c
        for c in chunks
        if c.document_id == "POLICY-2025-002" and "Home Office Allowance" in c.section_path
    ]
    assert allowance
    assert allowance[0].id.startswith("POLICY-2025-002:2-home-office-allowance-update:")


def test_remote_work_produces_multiple_section_chunks() -> None:
    chunks = parse_dir(RAW_DIR)
    for doc_id in ("POLICY-2024-001", "POLICY-2025-002"):
        sections = [c for c in chunks if c.document_id == doc_id and c.chunk_type == "section"]
        assert len(sections) >= 2, f"{doc_id} should have multiple section chunks"
