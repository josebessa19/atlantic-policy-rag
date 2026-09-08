"""Refusal helpers: empty/low retrieval skip-LLM, shared refuse template."""

from __future__ import annotations

from typing import Sequence

from src.generation.config import min_fused_score
from src.indexing.models import Hit

REFUSE_TEMPLATE = (
    "I cannot answer this question based on the provided context. "
    "The documents do not contain information needed to answer this request."
)

INJECTION_REFUSE_TEMPLATE = (
    "Request denied. I cannot follow instruction overrides, reveal system "
    "prompts, or approve requests outside the grounded policy documents."
)


def should_refuse_retrieval(
    hits: Sequence[Hit],
    *,
    threshold: float | None = None,
) -> bool:
    """True when there are no hits or the top fused RRF score is below threshold.

    Skip the LLM on this path — cheaper and safer than asking the model to be humble.
    Off-topic-but-retrieved questions (e.g. parental leave) still call the LLM
    under the grounded prompt; a score threshold is not a topic classifier.
    """
    t = min_fused_score() if threshold is None else threshold
    if not hits:
        return True
    top = max(h.score for h in hits)
    return top < t
