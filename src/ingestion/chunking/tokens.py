"""Cheap token estimation and BM25-friendly chunk text prefixes."""

from __future__ import annotations

import re
from typing import Literal


def estimate_tokens(text: str) -> int:
    """Approximate token count as ~4 characters per token (ceil)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def section_slug(section_path: str) -> str:
    """Lowercase hyphenated slug; empty path → ``body``."""
    if not section_path or not section_path.strip():
        return "body"
    slug = re.sub(r"[^a-z0-9]+", "-", section_path.lower()).strip("-")
    return slug or "body"


def format_bm25_prefix(
    document_id: str,
    status: Literal["active", "legacy"],
    section_label: str,
) -> str:
    """Prefix so BM25 can match document ids and section titles."""
    status_label = "Active" if status == "active" else "Legacy"
    return f"Document {document_id} ({status_label}) | Section: {section_label}"


def dense_index_text(chunk_text: str) -> str:
    """Body only for dense embeddings. BM25 / prompt text keep the prefix line."""
    raw = (chunk_text or "").strip()
    if not raw:
        return raw
    first, _, rest = raw.partition("\n")
    if first.startswith("Document ") and " | Section: " in first and rest.strip():
        return rest.strip()
    return raw


def embed_text(
    body: str,
    *,
    document_id: str,
    status: Literal["active", "legacy"],
    section_path: str,
    title: str | None = None,
) -> str:
    """Chunk text: BM25 prefix + body. Empty path uses document TITLE as section label."""
    body = body.strip()
    section_label = section_path.strip() if section_path.strip() else (title or "(document)")
    prefix = format_bm25_prefix(document_id, status, section_label)
    return f"{prefix}\n{body}" if body else prefix


def allocate_chunk_id(
    counters: dict[str, int],
    document_id: str,
    section_path: str,
) -> str:
    """Stable upsert key: ``{document_id}:{section_slug}:{n}`` (n per slug, document order)."""
    slug = section_slug(section_path)
    key = f"{document_id}:{slug}"
    n = counters.get(key, 0)
    counters[key] = n + 1
    return f"{document_id}:{slug}:{n}"
