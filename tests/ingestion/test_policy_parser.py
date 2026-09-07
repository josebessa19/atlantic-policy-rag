"""Policy header metadata: document_id, status, supersedes, effective_date."""

from datetime import date
from pathlib import Path

from src.ingestion.policy_parser import parse_policy_file

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def test_2024_legacy_metadata() -> None:
    doc = parse_policy_file(RAW_DIR / "Global_Remote_Work_Policy_2024.txt")
    assert doc.policy is not None
    assert doc.policy.document_id == "POLICY-2024-001"
    assert doc.policy.status == "legacy"
    assert doc.policy.effective_date == date(2024, 1, 1)
    assert doc.policy.supersedes == []
    assert doc.policy.title.startswith("Enterprise Remote Work Policy (2024")
    assert any(el.type == "heading" for el in doc.elements)


def test_2025_active_supersedes_2024() -> None:
    doc = parse_policy_file(RAW_DIR / "Global_Remote_Work_Policy_2025_Update.txt")
    assert doc.policy is not None
    assert doc.policy.document_id == "POLICY-2025-002"
    assert doc.policy.status == "active"
    assert doc.policy.effective_date == date(2025, 1, 1)
    assert doc.policy.supersedes == ["POLICY-2024-001"]
    # Numbered sections + bullets under section 1
    headings = [el for el in doc.elements if el.type == "heading"]
    assert len(headings) >= 2
    lists = [el for el in doc.elements if el.type == "list"]
    assert lists
    assert len(lists[0].items) == 3


def test_matrix_defaults_active_no_status() -> None:
    doc = parse_policy_file(RAW_DIR / "IT_Hardware_Allowance_Matrix.txt")
    assert doc.policy is not None
    assert doc.policy.document_id == "IT-SPEC-2025-A"
    assert doc.policy.status == "active"
    assert doc.policy.effective_date is None
    assert doc.policy.supersedes == []
    tables = [el for el in doc.elements if el.type == "table"]
    assert len(tables) == 1
    md = tables[0].markdown or ""
    assert "EMEA" in md
    assert "€1,800" in md
    assert "€60" in md


def test_no_pdf_ocr_dependency() -> None:
    """Ingest path is plain text — no Docling imports required."""
    import src.ingestion.policy_parser as pp

    assert not hasattr(pp, "DoclingParser")
    src = Path(pp.__file__).read_text(encoding="utf-8")
    assert "docling" not in src.lower()
    assert "import tesseract" not in src.lower()
