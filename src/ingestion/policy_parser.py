"""Parse Atlantic policy ``.txt`` files into ``ParsedDocument`` + lifecycle metadata.

When STATUS is missing (e.g. IT-SPEC-2025-A), default ``status=active``.
Corpus is born-digital text — no layout/PDF libraries on this path.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from src.ingestion.models import (
    DocumentMetadata,
    HeadingElement,
    ListElement,
    ParagraphElement,
    ParsedDocument,
    PolicyMetadata,
    TableElement,
)

PARSER_NAME = "policy_txt"

_HEADER_RE = re.compile(r"^([A-Z][A-Z0-9 _]*):\s*(.*)$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+)\.\s+(.+)$")
_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET_RE = re.compile(r"^-\s+(.+)$")
_SUPERSEDES_RE = re.compile(r"Supersedes\s+([A-Z0-9-]+)", re.IGNORECASE)
_TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|\s*$")

_HEADER_KEYS = {"DOCUMENT_ID", "TITLE", "EFFECTIVE DATE", "STATUS"}


def _parse_effective_date(raw: str | None) -> date | None:
    if not raw or not raw.strip():
        return None
    try:
        return datetime.strptime(raw.strip(), "%B %d, %Y").date()
    except ValueError:
        return None


def _map_status(raw: str | None) -> Literal["active", "legacy"]:
    if not raw:
        return "active"
    lower = raw.lower()
    if "legacy" in lower:
        return "legacy"
    if "active" in lower:
        return "active"
    return "active"


def _parse_supersedes(status_raw: str | None) -> list[str]:
    if not status_raw:
        return []
    return _SUPERSEDES_RE.findall(status_raw)


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 1


def _parse_table_row(line: str) -> list[str]:
    inner = line.strip().strip("|")
    return [cell.strip() for cell in inner.split("|")]


def _consume_table(lines: list[str], start: int) -> tuple[TableElement, int]:
    table_lines: list[str] = []
    i = start
    while i < len(lines) and (_is_table_line(lines[i]) or _TABLE_SEP_RE.match(lines[i].strip())):
        table_lines.append(lines[i].rstrip("\n"))
        i += 1
    markdown = "\n".join(table_lines)
    rows: list[list[str]] = []
    for tl in table_lines:
        if _TABLE_SEP_RE.match(tl.strip()):
            continue
        rows.append(_parse_table_row(tl))
    return TableElement(rows=rows, markdown=markdown), i


def _parse_header(lines: list[str]) -> tuple[dict[str, str], int]:
    fields: dict[str, str] = {}
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        if not raw.strip():
            i += 1
            break
        m = _HEADER_RE.match(raw.strip())
        if not m:
            break
        key = m.group(1).strip().upper()
        if key not in _HEADER_KEYS:
            break
        fields[key] = m.group(2).strip()
        i += 1
    return fields, i


def _parse_body(lines: list[str], start: int) -> list:
    elements: list = []
    i = start
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue

        numbered = _NUMBERED_HEADING_RE.match(stripped)
        if numbered:
            elements.append(HeadingElement(level=1, text=stripped))
            i += 1
            continue

        atx = _ATX_HEADING_RE.match(stripped)
        if atx:
            elements.append(HeadingElement(level=len(atx.group(1)), text=atx.group(2).strip()))
            i += 1
            continue

        if _is_table_line(stripped):
            table, i = _consume_table(lines, i)
            elements.append(table)
            continue

        bullet = _BULLET_RE.match(stripped)
        if bullet:
            items: list[str] = []
            while i < len(lines):
                b = _BULLET_RE.match(lines[i].strip())
                if not b:
                    break
                items.append(b.group(1).strip())
                i += 1
            elements.append(ListElement(ordered=False, items=items))
            continue

        elements.append(ParagraphElement(text=stripped))
        i += 1

    return elements


def parse_policy_text(text: str, *, source_path: str | Path) -> ParsedDocument:
    """Parse policy text into a structured document with lifecycle metadata."""
    lines = text.splitlines()
    fields, body_start = _parse_header(lines)

    document_id = fields.get("DOCUMENT_ID", "")
    if not document_id:
        raise ValueError(f"Missing DOCUMENT_ID in {source_path}")

    title = fields.get("TITLE", "")
    status_raw = fields.get("STATUS")
    policy = PolicyMetadata(
        document_id=document_id,
        title=title,
        effective_date=_parse_effective_date(fields.get("EFFECTIVE DATE")),
        status=_map_status(status_raw),
        supersedes=_parse_supersedes(status_raw),
    )

    elements = _parse_body(lines, body_start)
    return ParsedDocument(
        source_path=str(source_path),
        title=title or None,
        language="en",
        elements=elements,
        metadata=DocumentMetadata(parser_name=PARSER_NAME),
        policy=policy,
    )


def parse_policy_file(path: Path | str) -> ParsedDocument:
    """Read a UTF-8 policy ``.txt`` file and parse it."""
    path = Path(path)
    return parse_policy_text(path.read_text(encoding="utf-8"), source_path=path)
