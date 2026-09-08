"""Evaluation: deterministic benchmark helpers + optional RAGAS report."""

from src.eval.benchmark import (
    BenchmarkCase,
    assert_benchmark_case,
    load_benchmark,
)
from src.eval.runner import TracedResult, run_traced_query

__all__ = [
    "BenchmarkCase",
    "TracedResult",
    "assert_benchmark_case",
    "load_benchmark",
    "run_traced_query",
]
