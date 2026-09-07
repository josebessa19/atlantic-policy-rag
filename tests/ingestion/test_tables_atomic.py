"""Tables must stay atomic — never split across chunks."""

from src.ingestion.chunking import ChunkConfig, chunk_document, chunk_markdown
from src.ingestion.models import (
    DocumentMetadata,
    HeadingElement,
    ParagraphElement,
    ParsedDocument,
    TableElement,
)

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


def test_markdown_table_atomic() -> None:
    md = f"""# Policy

## Regional allowances

Some intro.

{TABLE_MD}

Example after table.
"""
    chunks = chunk_markdown(md, source="IT_Hardware_Allowance_Matrix.txt", config=ChunkConfig(max_tokens=40))
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "EMEA" in table_chunks[0].text
    assert "1200" in table_chunks[0].text
    for c in chunks:
        if c.chunk_type != "table" and "| EMEA |" in c.text:
            raise AssertionError("table row leaked into non-table chunk")
