from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rag.config import Settings
from rag.embeddings import OpenAIEmbedder


def test_embedding_order_batches_and_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-not-real")
    with patch("rag.embeddings.OpenAI") as client:
        response = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )
        client.return_value.embeddings.create.return_value = response
        embedder = OpenAIEmbedder(Settings())
        assert embedder.identity == "openai:text-embedding-3-small"
        assert embedder.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
        client.return_value.embeddings.create.reset_mock()
        embedder.embed(["text"] * 129)
        calls = client.return_value.embeddings.create.call_args_list
        assert len(calls) == 2
        assert len(calls[0].kwargs["input"]) == 128
        assert len(calls[1].kwargs["input"]) == 1
        client.return_value.embeddings.create.reset_mock()
        assert embedder.embed([]) == []
        client.return_value.embeddings.create.assert_not_called()
