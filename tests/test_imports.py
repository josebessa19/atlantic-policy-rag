"""Smoke test: core packages import cleanly after Step 0 scaffold."""


def test_core_packages_import():
    import src.ingestion  # noqa: F401
    import src.indexing  # noqa: F401
    import src.generation  # noqa: F401
    import src.guardrails  # noqa: F401
    import backend  # noqa: F401
