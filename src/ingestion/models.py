"""Parser-agnostic structured document model for ingestion."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"


class HeadingElement(BaseModel):
    type: Literal[ElementType.HEADING] = ElementType.HEADING
    level: int = Field(ge=1, le=6)
    text: str


class ParagraphElement(BaseModel):
    type: Literal[ElementType.PARAGRAPH] = ElementType.PARAGRAPH
    text: str


class TableElement(BaseModel):
    type: Literal[ElementType.TABLE] = ElementType.TABLE
    rows: list[list[str]] = Field(default_factory=list)
    markdown: str | None = None


class ListElement(BaseModel):
    type: Literal[ElementType.LIST] = ElementType.LIST
    ordered: bool = False
    items: list[str] = Field(default_factory=list)


DocumentElement = Annotated[
    HeadingElement | ParagraphElement | TableElement | ListElement,
    Field(discriminator="type"),
]


class DocumentMetadata(BaseModel):
    parser_name: str
    parser_version: str | None = None
    page_count: int | None = None
    extra: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    source_path: str
    title: str | None = None
    language: str | None = None
    elements: list[DocumentElement] = Field(default_factory=list)
    metadata: DocumentMetadata

    @classmethod
    def empty(cls, source_path: Path | str, parser_name: str) -> ParsedDocument:
        return cls(
            source_path=str(source_path),
            metadata=DocumentMetadata(parser_name=parser_name),
        )
