from pathlib import Path

import pytest

from rag.config import Settings
from rag.ingestion import ingest
from rag.models import Chunk
from rag.store import VectorStore


class FakeEmbedder:
    identity = "test:two-dimensional"
    calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


def test_cosine_search_and_persistence(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, "test")
    store.replace([Chunk("a", "a.txt", "text", "alpha")], [[8.0, 0.0]])
    store.replace([Chunk("b", "b.txt", "text", "beta")], [[0.0, 3.0]])
    hits = VectorStore(tmp_path, "test").search([2.0, 0.0], 10)
    assert hits[0].chunk.id == "a"
    assert hits[0].score == pytest.approx(1.0)
    assert store.sources() == ["a.txt", "b.txt"]


def test_idempotent_ingestion_and_replacement(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, FakeEmbedder.identity)
    embedder = FakeEmbedder()
    assert ingest("a.txt", b"old content", store, embedder, Settings()) == 1
    ingest("a.txt", b"old content", store, embedder, Settings())
    assert store.count == 1 and embedder.calls == 1
    ingest("a.txt", b"new content", store, embedder, Settings())
    assert store.count == 1 and embedder.calls == 2
    assert store.search([1.0, 0.0])[0].chunk.text == "new content"


def test_validation_does_not_destroy_existing_data(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, "test")
    chunk = Chunk("a", "a.txt", "text", "safe")
    store.replace([chunk], [[1.0, 0.0]])
    for vectors in ([], [[1.0]], [[float("nan"), 1.0]], [[0.0, 0.0]]):
        with pytest.raises(ValueError):
            store.replace([chunk], vectors)
        assert store.count == 1
    with pytest.raises(ValueError, match="model changed"):
        VectorStore(tmp_path, "other")
    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0])
    with pytest.raises(ValueError, match="positive"):
        store.search([1.0, 0.0], 0)


def test_empty_store(tmp_path: Path) -> None:
    store = VectorStore(tmp_path, "test")
    assert store.search([1.0]) == []
