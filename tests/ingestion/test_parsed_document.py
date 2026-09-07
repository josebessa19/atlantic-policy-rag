"""Unit tests for ParsedDocument model (no real parser required)."""

from pathlib import Path

from src.ingestion.models import (
    DocumentMetadata,
    HeadingElement,
    ListElement,
    ParagraphElement,
    ParsedDocument,
    TableElement,
)


def test_construct_fake_parsed_document() -> None:
    doc = ParsedDocument(
        source_path=str(Path("data/raw/Global_Remote_Work_Policy_2025_Update.txt")),
        title="Enterprise Remote Work Policy (2025 Update)",
        language="en",
        elements=[
            HeadingElement(level=1, text="Enterprise Remote Work Policy (2025 Update)"),
            ParagraphElement(text="This policy applies to eligible employees."),
            TableElement(
                rows=[
                    ["Region", "Laptop allowance (USD)", "Monitor allowance (USD)"],
                    ["EMEA", "1200", "300"],
                ],
                markdown=(
                    "| Region | Laptop allowance (USD) | Monitor allowance (USD) |\n"
                    "| --- | --- | --- |\n"
                    "| EMEA | 1200 | 300 |"
                ),
            ),
            ListElement(ordered=True, items=["Confirm eligibility", "Submit allowance request"]),
        ],
        metadata=DocumentMetadata(parser_name="fake", parser_version="0.0.1", page_count=1),
    )

    assert doc.title.startswith("Enterprise Remote Work")
    assert doc.elements[0].type == "heading"
    assert doc.elements[0].level == 1
    assert doc.elements[2].type == "table"
    assert doc.elements[2].rows[1][1] == "1200"
    assert doc.metadata.parser_name == "fake"

    restored = ParsedDocument.model_validate_json(doc.model_dump_json())
    assert len(restored.elements) == 4
    assert restored.elements[2].type == "table"


def test_empty_factory() -> None:
    doc = ParsedDocument.empty("x.txt", parser_name="stub")
    assert doc.elements == []
    assert doc.metadata.parser_name == "stub"
