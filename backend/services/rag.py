"""RAG orchestration: retrieve → temporal drop → guardrails → generate → cite.

Routers must not call Qdrant; all retrieval goes through this service.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

from backend.schemas.query import Citation, QueryResponse
from src.generation.gemini import generate_answer
from src.generation.prompts import hit_supports_answer, parse_generation
from src.guardrails.injection import is_injection_attempt, is_injection_output
from src.guardrails.refuse import (
    INJECTION_REFUSE_TEMPLATE,
    REFUSE_TEMPLATE,
    should_refuse_retrieval,
)
from src.guardrails.temporal import drop_superseded, load_active_supersedes
from src.indexing.config import context_limit
from src.indexing.models import Hit
from src.indexing.store import hybrid_search

logger = logging.getLogger(__name__)

UNAVAILABLE_TEMPLATE = (
    "I cannot answer this question right now. "
    "The policy service is temporarily unavailable."
)


def citations_from_hits(hits: Sequence[Hit]) -> list[Citation]:
    """Build citations from selected hits. Deduplicate by (document_id, section)."""
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for hit in hits:
        key = (hit.document_id, hit.section or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Citation(
                document_id=hit.document_id,
                section=hit.section or "",
                relevance_score=float(hit.score),
            )
        )
    return out


def take_context(hits: Sequence[Hit], *, limit: int | None = None) -> list[Hit]:
    """Keep the top-N remaining hits after temporal drop (Qdrant order = RRF desc)."""
    n = context_limit() if limit is None else limit
    return list(hits)[: max(int(n), 0)]


def citations_for_answer(
    answer: str,
    hits: Sequence[Hit],
    used_chunk_ids: Sequence[str],
) -> list[Citation]:
    """Cite model-reported chunk ids, else chunks whose distinctive spans appear in the answer."""
    by_id = {h.chunk_id: h for h in hits if h.chunk_id}
    selected: list[Hit] = []
    seen: set[str] = set()
    for cid in used_chunk_ids:
        hit = by_id.get(cid)
        if hit is None or cid in seen:
            continue
        seen.add(cid)
        selected.append(hit)
    if selected:
        return citations_from_hits(selected)
    supported = [h for h in hits if hit_supports_answer(h, answer)]
    return citations_from_hits(supported)


def answer_query(
    query: str,
    *,
    search_fn: Callable[[str], list[Hit]] | None = None,
    generate_fn: Callable[[str, list[Hit]], str] | None = None,
    active_supersedes: set[str] | None = None,
    load_supersedes_fn: Callable[[], set[str]] | None = None,
) -> QueryResponse:
    """Full pipeline. Injectable callables keep unit/API tests offline."""
    q = (query or "").strip()
    if not q:
        return QueryResponse(answer=REFUSE_TEMPLATE, refused=True, citations=[])

    if is_injection_attempt(q):
        return QueryResponse(
            answer=INJECTION_REFUSE_TEMPLATE,
            refused=True,
            citations=[],
        )

    search = search_fn or hybrid_search
    hits = list(search(q))

    if active_supersedes is not None:
        supersedes = active_supersedes
    elif load_supersedes_fn is not None:
        supersedes = load_supersedes_fn()
    else:
        supersedes = load_active_supersedes()

    after_drop = drop_superseded(hits, supersedes)
    kept = take_context(after_drop)

    top = max((h.score for h in kept), default=0.0)
    send = not should_refuse_retrieval(kept)
    logger.info(
        "retrieval raw=%d after_drop=%d context=%d top_score=%.4f send_gemini=%s ids=%s",
        len(hits),
        len(after_drop),
        len(kept),
        top,
        send,
        [h.document_id for h in kept],
    )

    if not send:
        return QueryResponse(answer=REFUSE_TEMPLATE, refused=True, citations=[])

    generate = generate_fn or generate_answer
    raw = generate(q, kept)

    if is_injection_output(raw):
        return QueryResponse(
            answer=INJECTION_REFUSE_TEMPLATE,
            refused=True,
            citations=[],
        )

    answer, used_ids, refused = parse_generation(raw)
    if refused:
        return QueryResponse(
            answer=answer or REFUSE_TEMPLATE,
            refused=True,
            citations=[],
        )

    return QueryResponse(
        answer=answer,
        refused=False,
        citations=citations_for_answer(answer, kept, used_ids),
    )
