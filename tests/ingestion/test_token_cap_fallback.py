"""Token-cap fallback: oversized sections become partial_section chunks."""

from src.ingestion.chunking import ChunkConfig, chunk_document
from src.ingestion.chunking.fallback import split_oversized_section
from src.ingestion.chunking.tokens import estimate_tokens
from src.ingestion.models import (
    DocumentMetadata,
    HeadingElement,
    ParagraphElement,
    ParsedDocument,
    TableElement,
)


def test_oversized_section_splits() -> None:
    paras = [f"Paragraph number {i}. " + ("word " * 40) for i in range(12)]
    body = "\n\n".join(paras)
    config = ChunkConfig(max_tokens=80, overlap_tokens=10)
    pieces = split_oversized_section(
        body,
        section_path="Policy > Long",
        source="synth.txt",
        config=config,
    )
    assert len(pieces) >= 2
    for p in pieces:
        assert p.chunk_type == "partial_section"
        assert p.section_path == "Policy > Long"
        assert estimate_tokens(p.text) <= config.max_tokens + 30


def test_table_not_split_by_fallback() -> None:
    huge_rows = [["ColA", "ColB"]] + [[f"r{i}a", f"r{i}b " + ("x" * 20)] for i in range(40)]
    md_lines = ["| ColA | ColB |", "| --- | --- |"]
    for a, b in huge_rows[1:]:
        md_lines.append(f"| {a} | {b} |")
    table_md = "\n".join(md_lines)

    doc = ParsedDocument(
        source_path="big_table.txt",
        elements=[
            HeadingElement(level=1, text="Data"),
            TableElement(rows=huge_rows, markdown=table_md),
            ParagraphElement(text="After table."),
        ],
        metadata=DocumentMetadata(parser_name="fake"),
    )
    chunks = chunk_document(doc, ChunkConfig(max_tokens=50))
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].text.count("|") > 10
