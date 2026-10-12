from pathlib import Path

import pytest

from rag.config import Settings, api_key


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DATA_DIR", "/tmp/custom-rag")
    monkeypatch.setenv("RAG_CHAT_MODEL", "test-model")
    assert Settings.from_env().data_dir == Path("/tmp/custom-rag")
    assert Settings.from_env().chat_model == "test-model"


def test_key_alias_and_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAL_API_KEY", "fake-alias")
    assert api_key() == "fake-alias"
    monkeypatch.setenv("OPENAI_API_KEY", "fake-standard")
    assert api_key() == "fake-standard"


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "OPENAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        api_key()
