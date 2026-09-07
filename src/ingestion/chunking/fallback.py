"""Recursive token-cap fallback for oversized section text (never tables)."""

from __future__ import annotations

import logging
import re
import uuid

from src.ingestion.chunking.models import Chunk, ChunkConfig
from src.ingestion.chunking.tokens import embed_text, estimate_tokens

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _overlap_suffix(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0 or not text:
        return ""
    char_budget = overlap_tokens * 4
    if len(text) <= char_budget:
        return text
    return text[-char_budget:].lstrip()


def _make_partial(
    body: str,
    *,
    section_path: str,
    source: str,
    page: int | None,
    element_index: int | None,
    id_prefix: str,
) -> Chunk:
    text = embed_text(section_path, body)
    return Chunk(
        id=f"{id_prefix}-{uuid.uuid4().hex[:8]}",
        text=text,
        section_path=section_path,
        source=source,
        chunk_type="partial_section",
        token_estimate=estimate_tokens(text),
        page=page,
        element_index=element_index,
    )


def _pack_units(
    units: list[str],
    *,
    section_path: str,
    source: str,
    config: ChunkConfig,
    page: int | None,
    element_index: int | None,
    id_prefix: str,
) -> list[Chunk]:
    """Greedy-pack units into chunks under max_tokens."""
    chunks: list[Chunk] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if not buf.strip():
            buf = ""
            return
        chunks.append(
            _make_partial(
                buf,
                section_path=section_path,
                source=source,
                page=page,
                element_index=element_index,
                id_prefix=id_prefix,
            )
        )
        overlap = _overlap_suffix(buf, config.overlap_tokens)
        buf = overlap

    for unit in units:
        if not unit.strip():
            continue
        # Single unit too large: try finer split or hard cut
        unit_embed = embed_text(section_path, unit)
        if estimate_tokens(unit_embed) > config.max_tokens:
            flush()
            sentences = _split_sentences(unit)
            if len(sentences) > 1:
                chunks.extend(
                    _pack_units(
                        sentences,
                        section_path=section_path,
                        source=source,
                        config=config,
                        page=page,
                        element_index=element_index,
                        id_prefix=id_prefix,
                    )
                )
            else:
                char_cap = max(config.max_tokens * 4, 1)
                for i in range(0, len(unit), char_cap):
                    part = unit[i : i + char_cap]
                    chunks.append(
                        _make_partial(
                            part,
                            section_path=section_path,
                            source=source,
                            page=page,
                            element_index=element_index,
                            id_prefix=id_prefix,
                        )
                    )
            buf = ""
            continue

        candidate = f"{buf}\n\n{unit}".strip() if buf.strip() else unit
        if estimate_tokens(embed_text(section_path, candidate)) > config.max_tokens and buf.strip():
            flush()
            buf = unit if not buf.strip() else f"{buf}\n\n{unit}".strip()
            # After flush, buf may be overlap + we need to set unit
            if estimate_tokens(embed_text(section_path, buf)) > config.max_tokens:
                # overlap alone + unit still too big: start fresh with unit
                buf = unit
        else:
            buf = candidate

    if buf.strip():
        chunks.append(
            _make_partial(
                buf,
                section_path=section_path,
                source=source,
                page=page,
                element_index=element_index,
                id_prefix=id_prefix,
            )
        )
    return chunks


def split_oversized_section(
    body: str,
    *,
    section_path: str,
    source: str,
    config: ChunkConfig,
    page: int | None = None,
    element_index: int | None = None,
    id_prefix: str = "partial",
) -> list[Chunk]:
    """Split section body under max_tokens; same section_path on every piece."""
    body = body.strip()
    if not body:
        return []

    full = embed_text(section_path, body)
    if estimate_tokens(full) <= config.max_tokens:
        return [
            Chunk(
                id=f"{id_prefix}-{uuid.uuid4().hex[:8]}",
                text=full,
                section_path=section_path,
                source=source,
                chunk_type="section",
                token_estimate=estimate_tokens(full),
                page=page,
                element_index=element_index,
            )
        ]

    units = _split_paragraphs(body)
    if len(units) <= 1:
        units = _split_sentences(body)
    if len(units) <= 1:
        char_cap = max(config.max_tokens * 4, 1)
        units = [body[i : i + char_cap] for i in range(0, len(body), char_cap)]

    return _pack_units(
        units,
        section_path=section_path,
        source=source,
        config=config,
        page=page,
        element_index=element_index,
        id_prefix=id_prefix,
    )


def warn_if_oversized_table(text: str, config: ChunkConfig, source: str) -> None:
    tok = estimate_tokens(text)
    if tok > config.max_tokens:
        logger.warning(
            "Table chunk exceeds max_tokens (%d > %d) for %s; keeping whole (atomic).",
            tok,
            config.max_tokens,
            source,
        )
