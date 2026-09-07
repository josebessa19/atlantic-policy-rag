"""Section path and path-prefix assertions."""

from src.ingestion.chunking import ChunkConfig, chunk_document, chunk_markdown
from src.ingestion.models import (
    DocumentMetadata,
    HeadingElement,
    ListElement,
    ParagraphElement,
    ParsedDocument,
    TableElement,
)


def test_structure_section_paths() -> None:
    doc = ParsedDocument(
        source_path="policy.txt",
        title="Policy",
        elements=[
            HeadingElement(level=1, text="Policy"),
            HeadingElement(level=2, text="Remote Work"),
            HeadingElement(level=3, text="Allowance"),
            ParagraphElement(text="Home office allowance is paid annually."),
            TableElement(rows=[["A", "B"], ["1", "2"]], markdown="| A | B |\n| --- | --- |\n| 1 | 2 |"),
            ListElement(items=["Step one", "Step two"]),
        ],
        metadata=DocumentMetadata(parser_name="fake"),
    )
    chunks = chunk_document(doc)
    sectionish = [c for c in chunks if c.chunk_type in ("section", "partial_section", "table")]
    assert sectionish
    for c in sectionish:
        assert c.section_path, f"empty path on {c.chunk_type}"
        assert c.text.startswith("["), f"missing path prefix: {c.text[:40]}"

    allowance_chunks = [c for c in chunks if "Allowance" in c.section_path]
    assert allowance_chunks
    assert any(c.section_path == "Policy > Remote Work > Allowance" for c in allowance_chunks)


def test_markdown_section_paths() -> None:
    md = """# Policy

## Remote Work

### Allowance

Home office allowance is paid annually.
"""
    chunks = chunk_markdown(md, source="policy.md", config=ChunkConfig())
    assert chunks
    assert any(c.section_path == "Policy > Remote Work > Allowance" for c in chunks)
    for c in chunks:
        if c.section_path:
            assert c.text.startswith(f"[{c.section_path}]")
