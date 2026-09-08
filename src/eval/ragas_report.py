"""Run RAGAS faithfulness + context recall on benchmark traces.

Contexts are the **post-temporal-drop** chunk texts Gemini saw
(``run_traced_query``), not raw hybrid hits.

Usage::

    pip install -e ".[eval]"
    python -m src.eval.ragas_report

Requires Qdrant ingested + ``GEMINI_API_KEY``. Report is gitignored.
Does **not** fail on score < 1.0 (LLM-as-judge variance).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.benchmark import BenchmarkCase, load_benchmark
from src.eval.runner import TracedResult, run_traced_query
from src.generation.config import gemini_api_key, llm_model

REPORT_MD = Path(__file__).resolve().parents[2] / "data" / "eval" / "ragas_report.md"
REPORT_JSON = Path(__file__).resolve().parents[2] / "data" / "eval" / "ragas_report.json"

# Injection never reaches generation — not a RAG quality sample.
SKIP_RAGAS = frozenset({"TEST-05"})
# Refusal reference is "not in docs" — context recall is N/A.
FAITHFULNESS_ONLY = frozenset({"TEST-03"})

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class RowScore:
    eval_id: str
    faithfulness: float | None
    context_recall: float | None
    note: str
    refused: bool
    n_contexts: int


def _require_ragas():
    try:
        from datasets import Dataset  # noqa: F401
        from ragas import evaluate  # noqa: F401
        from ragas.llms import LangchainLLMWrapper  # noqa: F401
        from ragas.metrics import context_recall, faithfulness  # noqa: F401
        from langchain_openai import ChatOpenAI  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "RAGAS deps missing. Install with: pip install -e \".[eval]\"\n"
            f"Import error: {exc}"
        ) from exc


def _build_judge_llm():
    """Gemini Flash via OpenAI-compatible endpoint (avoids instructor/genai quirk)."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    key = gemini_api_key()
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key."
        )
    chat = ChatOpenAI(
        model=llm_model(),
        api_key=key,
        base_url=GEMINI_OPENAI_BASE,
        temperature=0.0,
        max_retries=3,
    )
    return LangchainLLMWrapper(chat)


def _transient_gemini_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(
        token in text
        for token in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")
    )


def _collect_traces(cases: list[BenchmarkCase]) -> dict[str, TracedResult]:
    out: dict[str, TracedResult] = {}
    for case in cases:
        print(f"  tracing {case.eval_id} …", flush=True)
        for attempt in range(4):
            try:
                out[case.eval_id] = run_traced_query(case.query)
                time.sleep(2)  # breathe between free-tier generation calls
                break
            except Exception as exc:  # noqa: BLE001 — retry quota / blips
                if attempt == 3 or not _transient_gemini_error(exc):
                    raise
                wait = 20 * (attempt + 1)
                print(f"    transient Gemini error, retry in {wait}s: {exc}", flush=True)
                time.sleep(wait)
    return out


def _score_with_ragas(
    cases: list[BenchmarkCase],
    traces: dict[str, TracedResult],
) -> list[RowScore]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_recall, faithfulness

    llm = _build_judge_llm()
    rows: list[RowScore] = []

    for case in cases:
        traced = traces[case.eval_id]
        resp = traced.response
        n_ctx = len(traced.retrieved_contexts)

        if case.eval_id in SKIP_RAGAS:
            rows.append(
                RowScore(
                    eval_id=case.eval_id,
                    faithfulness=None,
                    context_recall=None,
                    note=(
                        "Skipped: injection refused before LLM; empty context "
                        "is not a RAG quality sample (pytest is the contract)."
                    ),
                    refused=resp.refused,
                    n_contexts=n_ctx,
                )
            )
            continue

        sample = {
            "user_input": case.query,
            "response": resp.answer,
            "retrieved_contexts": traced.retrieved_contexts or [""],
            "reference": case.expected_ground_truth,
        }

        if case.eval_id in FAITHFULNESS_ONLY:
            print(f"  RAGAS faithfulness only: {case.eval_id}", flush=True)
            ds = Dataset.from_list([sample])
            result = evaluate(
                ds,
                metrics=[faithfulness],
                llm=llm,
                raise_exceptions=True,
            )
            scores = _result_to_dict(result)
            rows.append(
                RowScore(
                    eval_id=case.eval_id,
                    faithfulness=_as_float(scores.get("faithfulness")),
                    context_recall=None,
                    note=(
                        "Context recall N/A: reference is a refusal "
                        "('not in the documents'), not a policy fact to retrieve. "
                        "Faithfulness checks the refusal is grounded (no invented leave)."
                    ),
                    refused=resp.refused,
                    n_contexts=n_ctx,
                )
            )
            continue

        print(f"  RAGAS faithfulness+recall: {case.eval_id}", flush=True)
        ds = Dataset.from_list([sample])
        result = evaluate(
            ds,
            metrics=[faithfulness, context_recall],
            llm=llm,
            raise_exceptions=True,
        )
        scores = _result_to_dict(result)
        rows.append(
            RowScore(
                eval_id=case.eval_id,
                faithfulness=_as_float(scores.get("faithfulness")),
                context_recall=_as_float(scores.get("context_recall")),
                note="Post-drop contexts only (Option A).",
                refused=resp.refused,
                n_contexts=n_ctx,
            )
        )

    return rows


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        if len(df) == 1:
            return {c: df.iloc[0][c] for c in df.columns}
    if isinstance(result, dict):
        return result
    # EvaluationResult-like
    scores = getattr(result, "scores", None)
    if scores is not None and len(scores) == 1:
        return dict(scores[0])
    return dict(result)  # type: ignore[arg-type]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def write_report(rows: list[RowScore], path_md: Path, path_json: Path) -> None:
    lines = [
        "# RAGAS report — Atlantic Policy RAG",
        "",
        "Judge: Gemini Flash via OpenAI-compatible endpoint (same `GEMINI_API_KEY`).",
        "Contexts: **post `drop_superseded`** chunk texts (same as the LLM prompt).",
        "Metrics: **faithfulness** + **context recall** only (LLM-only; no Google embeddings).",
        "",
        "**Pytest is the contract.** RAGAS scores are a dashboard — LLM-as-judge variance",
        "is expected; do not tune the pipeline until every score is 1.0.",
        "",
        "| eval_id | faithfulness | context_recall | refused | n_contexts | note |",
        "|---------|--------------|----------------|---------|------------|------|",
    ]
    for r in rows:
        note = r.note.replace("|", "/")
        lines.append(
            f"| {r.eval_id} | {_fmt_score(r.faithfulness)} | "
            f"{_fmt_score(r.context_recall)} | {r.refused} | {r.n_contexts} | {note} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **TEST-03:** context recall N/A — ground truth is a refusal, not a fact to retrieve.",
            "- **TEST-05:** skipped — injection path never calls the generator.",
            "- Low faithfulness with high pytest pass → wording variance; investigate only if claims invent policy.",
            "- Low context recall with pytest fail on citations → retrieval / temporal drop issue (fix pipeline, not the judge).",
            "",
        ]
    )
    path_md.parent.mkdir(parents=True, exist_ok=True)
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = [
        {
            "eval_id": r.eval_id,
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "refused": r.refused,
            "n_contexts": r.n_contexts,
            "note": r.note,
        }
        for r in rows
    ]
    path_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _ = argv
    _require_ragas()
    cases = load_benchmark()
    print("Collecting live traces (Qdrant + Gemini)…", flush=True)
    traces = _collect_traces(cases)
    print("Scoring with RAGAS…", flush=True)
    rows = _score_with_ragas(cases, traces)
    write_report(rows, REPORT_MD, REPORT_JSON)
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_JSON}")
    for r in rows:
        print(
            f"  {r.eval_id}: faithfulness={_fmt_score(r.faithfulness)} "
            f"context_recall={_fmt_score(r.context_recall)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
