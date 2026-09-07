"""Markdown-section chunker unit tests."""

from src.ingestion.chunking import ChunkConfig, chunk_markdown


def test_markdown_splits_on_headings() -> None:
    md = """# Title

Intro paragraph.

## Section A

Body A with enough words.

## Section B

Body B.
"""
    chunks = chunk_markdown(md, source="doc.md", config=ChunkConfig())
    paths = {c.section_path for c in chunks}
    assert "Title" in paths or any("Title" in p for p in paths)
    assert any("Section A" in p for p in paths)
    assert any("Section B" in p for p in paths)
