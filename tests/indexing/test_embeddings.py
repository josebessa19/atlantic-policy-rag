"""Unit tests for local BGE-M3 embedding client (mocked sentence-transformers)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.indexing import embeddings as emb_mod
from src.indexing.config import DEFAULT_EMBEDDING_DIM


@pytest.fixture(autouse=True)
def _reset_model():
    emb_mod.reset_embedding_model()
    yield
    emb_mod.reset_embedding_model()


def test_embed_texts_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        emb_mod.embed_texts([])


def test_embed_texts_rejects_blank_string() -> None:
    with pytest.raises(ValueError, match="empty text"):
        emb_mod.embed_texts(["ok", "  "])


def test_embed_query_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        emb_mod.embed_query("")


def test_embed_query_uses_bge_prefix() -> None:
    dim = DEFAULT_EMBEDDING_DIM
    fake_vec = [0.1] * dim
    mock_model = MagicMock()
    mock_model.encode.return_value = [fake_vec]

    with patch.object(emb_mod, "_get_model", return_value=mock_model):
        emb_mod.embed_query("laptop Germany")

    sent = mock_model.encode.call_args.args[0][0]
    assert sent.startswith(emb_mod.QUERY_PREFIX)
    assert sent.endswith("laptop Germany")


def test_embed_texts_returns_expected_dim() -> None:
    dim = DEFAULT_EMBEDDING_DIM
    fake_vec = [0.1] * dim

    mock_model = MagicMock()
    mock_model.encode.return_value = [fake_vec, fake_vec]

    with patch.object(emb_mod, "_get_model", return_value=mock_model):
        out = emb_mod.embed_texts(["a", "b"])

    assert len(out) == 2
    assert all(len(v) == dim for v in out)
    mock_model.encode.assert_called_once()


def test_embed_texts_rejects_wrong_length() -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2]]

    with patch.object(emb_mod, "_get_model", return_value=mock_model):
        with pytest.raises(ValueError, match="length"):
            emb_mod.embed_texts(["a"])


def test_embed_texts_rejects_all_zeros() -> None:
    dim = DEFAULT_EMBEDDING_DIM
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.0] * dim]

    with patch.object(emb_mod, "_get_model", return_value=mock_model):
        with pytest.raises(ValueError, match="all-zero"):
            emb_mod.embed_texts(["a"])


@pytest.mark.live
def test_live_bge_m3_dim() -> None:
    if os.getenv("RUN_LIVE_EMBEDDING") != "1":
        pytest.skip("Set RUN_LIVE_EMBEDDING=1 to download/run BGE-M3")
    vec = emb_mod.embed_query("laptop allowance Germany")
    assert len(vec) == DEFAULT_EMBEDDING_DIM
    assert any(v != 0.0 for v in vec)
