"""Section path and BM25 prefix assertions."""

from src.ingestion.chunking import ChunkConfig, chunk_document
from src.ingestion.models import (
    DocumentMetadata,
    HeadingElement,
    ListElement,
    ParagraphElement,
    ParsedDocument,
    PolicyMetadata,
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
        policy=PolicyMetadata(document_id="POLICY-TEST", title="Policy", status="active"),
    )
    chunks = chunk_document(doc)
    sectionish = [c for c in chunks if c.chunk_type in ("section", "partial_section", "table")]
    assert sectionish
    for c in sectionish:
        assert c.section_path, f"empty path on {c.chunk_type}"
        assert c.text.startswith("Document POLICY-TEST (Active) | Section:"), c.text[:60]

    allowance_chunks = [c for c in chunks if "Allowance" in c.section_path]
    assert allowance_chunks
    assert any(c.section_path == "Policy > Remote Work > Allowance" for c in allowance_chunks)
