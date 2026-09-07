"""Tables must stay atomic — never split across chunks."""

from pathlib import Path

from src.ingestion.chunking import ChunkConfig, chunk_document
from src.ingestion.models import (
    DocumentMetadata,
    HeadingElement,
    ParagraphElement,
    ParsedDocument,
    PolicyMetadata,
    TableElement,
)
from src.ingestion.pipeline import parse_dir

TABLE_MD = (
    "| Region | Laptop allowance (USD) | Monitor allowance (USD) |\n"
    "| --- | --- | --- |\n"
    "| Americas | 1500 | 400 |\n"
    "| EMEA | 1200 | 300 |\n"
    "| APAC | 1100 | 250 |\n"
    "| LATAM | 900 | 200 |"
)

TABLE_ROWS = [
    ["Region", "Laptop allowance (USD)", "Monitor allowance (USD)"],
    ["Americas", "1500", "400"],
    ["EMEA", "1200", "300"],
    ["APAC", "1100", "250"],
    ["LATAM", "900", "200"],
]

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _hardware_doc() -> ParsedDocument:
    return ParsedDocument(
        source_path="data/raw/IT_Hardware_Allowance_Matrix.txt",
        title="IT Hardware Allowance Matrix",
        elements=[
            HeadingElement(level=1, text="IT Hardware Allowance Matrix"),
            HeadingElement(level=2, text="Regional allowances"),
            ParagraphElement(text="Allowances vary by region."),
            TableElement(rows=TABLE_ROWS, markdown=TABLE_MD),
            ParagraphElement(text="Example: EMEA laptop allowance is 1200 USD."),
        ],
        metadata=DocumentMetadata(parser_name="fake"),
        policy=PolicyMetadata(document_id="IT-SPEC-2025-A", title="IT Hardware Allowance Matrix"),
    )


def test_structure_table_atomic() -> None:
    chunks = chunk_document(_hardware_doc(), ChunkConfig(max_tokens=50))
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) >= 1
    text = table_chunks[0].text
    assert "EMEA" in text
    assert "1200" in text
    assert "LATAM" in text
    assert text.count("EMEA") == 1
    for c in table_chunks:
        assert "Americas" in c.text or "Region" in c.text


def test_real_matrix_table_atomic_with_euro_figures() -> None:
    """Hardware matrix: one table chunk with all regions + EMEA euro amounts."""
    chunks = parse_dir(RAW_DIR)
    matrix = [c for c in chunks if c.document_id == "IT-SPEC-2025-A" and c.chunk_type == "table"]
    assert len(matrix) == 1
    text = matrix[0].text
    assert "EMEA" in text
    assert "US / Canada" in text
    assert "€1,800" in text
    assert "€60" in text
    assert "APAC" in text
