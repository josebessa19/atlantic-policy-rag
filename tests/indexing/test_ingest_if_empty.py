"""Unit tests for empty-collection ingest (mocked — no GPU / Qdrant)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.indexing.ingest import collection_is_empty, ingest_if_empty


def test_collection_is_empty_when_missing() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False
    assert collection_is_empty(client=client, collection="atlantic_policies") is True


def test_collection_is_empty_when_zero_points() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    info = MagicMock()
    info.points_count = 0
    client.get_collection.return_value = info
    assert collection_is_empty(client=client, collection="atlantic_policies") is True


def test_collection_not_empty_when_points_exist() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    info = MagicMock()
    info.points_count = 12
    client.get_collection.return_value = info
    assert collection_is_empty(client=client, collection="atlantic_policies") is False


def test_ingest_if_empty_skips_when_populated() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    info = MagicMock()
    info.points_count = 5
    client.get_collection.return_value = info

    with patch("src.indexing.ingest.index_chunks") as index_chunks:
        n = ingest_if_empty(data_dir=Path("data/raw"), client=client)
    assert n == 0
    index_chunks.assert_not_called()


def test_ingest_if_empty_indexes_when_empty(tmp_path: Path) -> None:
    client = MagicMock()
    client.collection_exists.return_value = False
    raw = tmp_path / "raw"
    raw.mkdir()

    fake_chunks = [MagicMock()]
    with (
        patch("src.indexing.ingest.parse_dir", return_value=fake_chunks) as parse_dir,
        patch("src.indexing.ingest.index_chunks", return_value=1) as index_chunks,
    ):
        n = ingest_if_empty(data_dir=raw, client=client, collection="test_coll")

    assert n == 1
    parse_dir.assert_called_once_with(raw)
    index_chunks.assert_called_once()
    assert index_chunks.call_args.kwargs["collection"] == "test_coll"
    assert index_chunks.call_args.kwargs["recreate"] is False


def test_auto_ingest_default_off(monkeypatch) -> None:
    from src.indexing.config import auto_ingest

    monkeypatch.delenv("AUTO_INGEST", raising=False)
    assert auto_ingest() is False
    monkeypatch.setenv("AUTO_INGEST", "1")
    assert auto_ingest() is True
    monkeypatch.setenv("AUTO_INGEST", "true")
    assert auto_ingest() is True
    monkeypatch.setenv("AUTO_INGEST", "0")
    assert auto_ingest() is False
