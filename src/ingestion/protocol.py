"""Document parser protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.ingestion.models import ParsedDocument


@runtime_checkable
class DocumentParser(Protocol):
    """Parse a document file into a normalized ParsedDocument."""

    name: str

    def parse(self, path: Path) -> ParsedDocument:
        """Parse ``path`` into a structured document model."""
        ...
