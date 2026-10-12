import argparse
from pathlib import Path

from rag.config import Settings
from rag.documents import chunk_sections, load_document
from rag.embeddings import Embedder, OpenAIEmbedder
from rag.store import VectorStore


def ingest(
    filename: str, data: bytes, store: VectorStore, embedder: Embedder, settings: Settings
) -> int:
    if store.identity != embedder.identity:
        raise ValueError("Store and embedder must use the same model.")
    chunks = chunk_sections(
        load_document(filename, data), settings.chunk_tokens, settings.overlap_tokens
    )
    if not chunks:
        raise ValueError("Document has no indexable text.")
    if not store.contains(chunks):
        store.replace(chunks, embedder.embed([chunk.text for chunk in chunks]))
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF, DOCX, TXT, or MD files locally.")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    embedder = OpenAIEmbedder(settings)
    store = VectorStore(settings.data_dir, embedder.identity)
    for path in args.files:
        count = ingest(path.name, path.read_bytes(), store, embedder, settings)
        print(f"{path.name}: {count} chunks indexed ({store.count} total)")


if __name__ == "__main__":
    main()
