"""Run ``answer_query`` while capturing post-temporal-drop hits.

``generate_fn`` receives *kept* hits only (after ``drop_superseded``). Wrapping
the real generator lets RAGAS / pytest see the same chunk texts Gemini saw,
without changing the production ``answer_query`` signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from backend.schemas.query import QueryResponse
from backend.services.rag import answer_query
from src.generation.gemini import generate_answer
from src.indexing.models import Hit


@dataclass
class TracedResult:
    """Pipeline response plus the contexts actually passed to generation."""

    response: QueryResponse
    kept_hits: list[Hit] = field(default_factory=list)
    generate_called: bool = False

    @property
    def retrieved_contexts(self) -> list[str]:
        """Chunk texts for RAGAS — same bodies as in ``build_user_message``."""
        return [h.text for h in self.kept_hits if h.text]


def run_traced_query(
    query: str,
    *,
    search_fn: Callable[[str], list[Hit]] | None = None,
    generate_fn: Callable[[str, list[Hit]], str] | None = None,
    active_supersedes: set[str] | None = None,
    load_supersedes_fn: Callable[[], set[str]] | None = None,
) -> TracedResult:
    """Call ``answer_query`` and capture kept hits from the generate path.

    Injection / empty / low-score refusals never call generate → ``kept_hits``
    stays empty (correct for TEST-05 and skip-LLM paths).
    """
    captured: list[Hit] = []
    called = {"generate": False}
    inner = generate_fn or generate_answer

    def tracing_generate(q: str, hits: list[Hit]) -> str:
        called["generate"] = True
        captured.clear()
        captured.extend(hits)
        return inner(q, hits)

    response = answer_query(
        query,
        search_fn=search_fn,
        generate_fn=tracing_generate,
        active_supersedes=active_supersedes,
        load_supersedes_fn=load_supersedes_fn,
    )
    return TracedResult(
        response=response,
        kept_hits=list(captured),
        generate_called=called["generate"],
    )
