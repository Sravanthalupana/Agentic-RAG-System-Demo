import sqlite3
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from rag.models import Chunk, Hit


def normalized(vectors: list[list[float]]) -> NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must be finite, rectangular, and nonempty.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embedding vectors must be nonzero.")
    return np.asarray(matrix / norms, dtype=np.float32)


class VectorStore:
    """SQLite persists source/vector pairs atomically; FAISS performs cosine search."""

    def __init__(self, directory: Path, embedding_identity: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "vectors.sqlite3"
        self.identity = embedding_identity
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, source TEXT, "
                "location TEXT, text TEXT, vector BLOB)"
            )
            existing = connection.execute("SELECT value FROM meta WHERE key='embedding'").fetchone()
            if existing and existing[0] != self.identity:
                raise ValueError("Embedding model changed; choose a separate data directory.")
            connection.execute(
                "INSERT OR IGNORE INTO meta VALUES ('embedding', ?)", (self.identity,)
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @property
    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT count(*) FROM chunks").fetchone()[0])

    def sources(self) -> list[str]:
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute("SELECT DISTINCT source FROM chunks ORDER BY source")
            ]

    def contains(self, chunks: list[Chunk]) -> bool:
        if not chunks:
            return False
        with self._connect() as connection:
            existing = {
                row[0]
                for row in connection.execute(
                    "SELECT id FROM chunks WHERE source=?", (chunks[0].source,)
                )
            }
        return existing == {chunk.id for chunk in chunks}

    def replace(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("One embedding is required per chunk.")
        if len({chunk.source for chunk in chunks}) != 1:
            raise ValueError("Replace one source at a time.")
        matrix = normalized(vectors)
        with self._connect() as connection:
            dimension = connection.execute(
                "SELECT value FROM meta WHERE key='dimension'"
            ).fetchone()
            if dimension and int(dimension[0]) != matrix.shape[1]:
                raise ValueError("Embedding dimension mismatch.")
            connection.execute(
                "INSERT OR IGNORE INTO meta VALUES ('dimension', ?)", (str(matrix.shape[1]),)
            )
            connection.execute("DELETE FROM chunks WHERE source=?", (chunks[0].source,))
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (chunk.id, chunk.source, chunk.location, chunk.text, vector.tobytes())
                    for chunk, vector in zip(chunks, matrix, strict=True)
                ],
            )

    def search(self, vector: list[float], top_k: int = 4) -> list[Hit]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, source, location, text, vector FROM chunks ORDER BY id"
            ).fetchall()
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
        query = normalized([vector])
        if query.shape[1] != matrix.shape[1]:
            raise ValueError("Query embedding dimension mismatch.")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        scores, indices = index.search(query, min(top_k, len(rows)))
        return [
            Hit(Chunk(*rows[int(i)][:4]), float(score))
            for score, i in zip(scores[0], indices[0], strict=True)
        ]
