"""RAG orchestration: retrieve → temporal drop → guardrails → generate → cite.

Routers must not call Qdrant; all retrieval goes through this service.
"""

from __future__ import annotations

from typing import Callable, Sequence

from backend.schemas.query import Citation, QueryResponse
from src.generation.gemini import generate_answer
from src.generation.prompts import assemble_context, looks_like_refusal
from src.guardrails.injection import is_injection_attempt
from src.guardrails.refuse import (
    INJECTION_REFUSE_TEMPLATE,
    REFUSE_TEMPLATE,
    should_refuse_retrieval,
)
from src.guardrails.temporal import drop_superseded, load_active_supersedes
from src.indexing.models import Hit
from src.indexing.store import hybrid_search


def citations_from_hits(hits: Sequence[Hit]) -> list[Citation]:
    """Build citations only from hits actually passed to the prompt.

    Deduplicate by (document_id, section). relevance_score = fused Qdrant RRF score.
    """
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

    kept = drop_superseded(hits, supersedes)

    # Side effect for tests / debugging: assembler string without 2024 after drop.
    _ = assemble_context(kept)

    if should_refuse_retrieval(kept):
        return QueryResponse(answer=REFUSE_TEMPLATE, refused=True, citations=[])

    generate = generate_fn or generate_answer
    answer = generate(q, kept)

    if looks_like_refusal(answer):
        return QueryResponse(answer=answer, refused=True, citations=[])

    return QueryResponse(
        answer=answer,
        refused=False,
        citations=citations_from_hits(kept),
    )
