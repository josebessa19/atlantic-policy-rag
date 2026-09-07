"""Markdown-section chunker: split on ATX headings; keep pipe tables atomic."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from src.ingestion.chunking.fallback import split_oversized_section, warn_if_oversized_table
from src.ingestion.chunking.models import Chunk, ChunkConfig
from src.ingestion.chunking.tokens import embed_text, estimate_tokens

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|\s*$")


def _new_id(kind: str) -> str:
    return f"{kind}-{uuid.uuid4().hex[:8]}"


def _path_string(stack: list[str]) -> str:
    return " > ".join(stack)


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|")


def chunk_markdown(
    markdown: str,
    *,
    source: str,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Split Markdown on headings; keep pipe tables as single chunks."""
    config = config or ChunkConfig()
    source_name = Path(source).name if source else "unknown.md"
    lines = markdown.splitlines()

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def current_path() -> str:
        return _path_string([t for _, t in heading_stack])

    def flush_section() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        buffer = []
        if not body:
            return
        pieces = split_oversized_section(
            body,
            section_path=current_path(),
            source=source_name,
            config=config,
            id_prefix="md-sec",
        )
        chunks.extend(pieces)

    def emit_table(table_lines: list[str]) -> None:
        flush_section()
        table_body = "\n".join(table_lines).strip()
        if not table_body:
            return
        path = current_path()
        embedded = embed_text(path, table_body)
        warn_if_oversized_table(embedded, config, source_name)
        chunks.append(
            Chunk(
                id=_new_id("md-table"),
                text=embedded,
                section_path=path,
                source=source_name,
                chunk_type="table",
                token_estimate=estimate_tokens(embedded),
            )
        )

    i = 0
    while i < len(lines):
        line = lines[i]
        heading_m = _HEADING_RE.match(line)
        if heading_m:
            flush_section()
            level = len(heading_m.group(1))
            title = heading_m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            i += 1
            continue

        if _is_table_line(line):
            table_lines = [line]
            i += 1
            while i < len(lines) and (_is_table_line(lines[i]) or _TABLE_SEP_RE.match(lines[i])):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                emit_table(table_lines)
            else:
                buffer.extend(table_lines)
            continue

        buffer.append(line)
        i += 1

    flush_section()
    return chunks
