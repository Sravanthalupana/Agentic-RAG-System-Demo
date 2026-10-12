import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(".data")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_tokens: int = 500
    overlap_tokens: int = 75
    top_k: int = 4
    max_retrievals: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("RAG_DATA_DIR", ".data")),
            chat_model=os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        )


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAL_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY (or OPENAL_API_KEY), or select Offline demo.")
    return key
