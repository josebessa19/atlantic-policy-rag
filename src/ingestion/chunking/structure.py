"""Structure-aware chunker: walk ParsedDocument typed elements."""

from __future__ import annotations

import uuid
from pathlib import Path

from src.ingestion.chunking.fallback import split_oversized_section, warn_if_oversized_table
from src.ingestion.chunking.models import Chunk, ChunkConfig
from src.ingestion.chunking.tokens import embed_text, estimate_tokens
from src.ingestion.models import (
    HeadingElement,
    ListElement,
    ParagraphElement,
    ParsedDocument,
    TableElement,
)


def _source_name(doc: ParsedDocument) -> str:
    return Path(doc.source_path).name


def _path_string(stack: list[str]) -> str:
    return " > ".join(stack)


def _render_list(el: ListElement) -> str:
    lines: list[str] = []
    for idx, item in enumerate(el.items, start=1):
        prefix = f"{idx}." if el.ordered else "-"
        lines.append(f"{prefix} {item}")
    return "\n".join(lines)


def _render_table(el: TableElement) -> str:
    if el.markdown and el.markdown.strip():
        return el.markdown.strip()
    if not el.rows:
        return ""
    lines = ["| " + " | ".join(row) + " |" for row in el.rows]
    if len(lines) >= 1:
        cols = len(el.rows[0])
        sep = "| " + " | ".join("---" for _ in range(cols)) + " |"
        return "\n".join([lines[0], sep, *lines[1:]])
    return "\n".join(lines)


def _new_id(kind: str) -> str:
    return f"{kind}-{uuid.uuid4().hex[:8]}"


def chunk_structure(
    doc: ParsedDocument,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Walk typed elements; heading path; atomic tables."""
    config = config or ChunkConfig()
    source = _source_name(doc)
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    buffer_parts: list[str] = []
    buffer_start_index: int | None = None

    def current_path() -> str:
        return _path_string([t for _, t in heading_stack])

    def flush_section() -> None:
        nonlocal buffer_parts, buffer_start_index
        body = "\n\n".join(p for p in buffer_parts if p.strip()).strip()
        buffer_parts = []
        start_idx = buffer_start_index
        buffer_start_index = None
        if not body:
            return
        pieces = split_oversized_section(
            body,
            section_path=current_path(),
            source=source,
            config=config,
            element_index=start_idx,
            id_prefix="sec",
        )
        chunks.extend(pieces)

    for idx, el in enumerate(doc.elements):
        if isinstance(el, HeadingElement):
            flush_section()
            while heading_stack and heading_stack[-1][0] >= el.level:
                heading_stack.pop()
            heading_stack.append((el.level, el.text.strip()))
            continue

        if isinstance(el, TableElement):
            flush_section()
            table_body = _render_table(el)
            if not table_body:
                continue
            path = current_path()
            text = embed_text(path, table_body)
            warn_if_oversized_table(text, config, source)
            chunks.append(
                Chunk(
                    id=_new_id("table"),
                    text=text,
                    section_path=path,
                    source=source,
                    chunk_type="table",
                    token_estimate=estimate_tokens(text),
                    element_index=idx,
                )
            )
            continue

        if isinstance(el, ParagraphElement):
            if buffer_start_index is None:
                buffer_start_index = idx
            buffer_parts.append(el.text.strip())
            continue

        if isinstance(el, ListElement):
            if buffer_start_index is None:
                buffer_start_index = idx
            buffer_parts.append(_render_list(el))
            continue

    flush_section()
    return chunks
