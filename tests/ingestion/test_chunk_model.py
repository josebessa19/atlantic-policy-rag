"""Unit tests for Chunk / ChunkConfig model."""

from src.ingestion.chunking import Chunk, ChunkConfig, chunk_document
from src.ingestion.models import DocumentMetadata, HeadingElement, ParagraphElement, ParsedDocument


def test_chunk_config_defaults() -> None:
    cfg = ChunkConfig()
    assert cfg.max_tokens == 500
    assert cfg.overlap_tokens == 50
    assert cfg.min_tokens is None


def test_chunk_model_fields() -> None:
    c = Chunk(
        id="POLICY-2025-002:workplace-flexibility:0",
        text="Document POLICY-2025-002 (Active) | Section: Workplace Flexibility\nHello",
        section_path="Workplace Flexibility",
        source="Global_Remote_Work_Policy_2025_Update.txt",
        chunk_type="section",
        token_estimate=10,
        document_id="POLICY-2025-002",
        status="active",
    )
    assert c.chunk_type == "section"
    assert c.section_path == "Workplace Flexibility"
    assert c.document_id == "POLICY-2025-002"
    dumped = c.model_dump()
    assert dumped["text"].startswith("Document POLICY-2025-002 (Active)")


def test_chunk_document_signature_structure() -> None:
    doc = ParsedDocument(
        source_path="x.txt",
        title="T",
        elements=[
            HeadingElement(level=1, text="Intro"),
            ParagraphElement(text="Short body."),
        ],
        metadata=DocumentMetadata(parser_name="fake"),
    )
    chunks = chunk_document(doc, ChunkConfig())
    assert len(chunks) >= 1
    assert chunks[0].section_path == "Intro"
    assert chunks[0].text.startswith("Document x (Active) | Section: Intro")
    assert chunks[0].id == "x:intro:0"
