from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, RateLimitError

from rag.config import Settings
from rag.runtime import create_agent, safe_error


def test_demo_uses_separate_store(tmp_path: Path) -> None:
    agent = create_agent("Offline demo", Settings(data_dir=tmp_path))
    assert agent.store.path.parent == tmp_path / "offline"
    assert agent.ask("Hello").retrieval_count == 0


def test_safe_error_does_not_expose_provider_body() -> None:
    response = httpx.Response(401, request=httpx.Request("POST", "https://example.com"))
    error = AuthenticationError("sensitive provider body", response=response, body=None)
    assert "sensitive" not in safe_error(error)
    assert "authentication" in safe_error(error)
    error = RateLimitError("sensitive provider body", response=response, body=None)
    assert "quota" in safe_error(error)
    assert "sensitive" not in safe_error(ValueError("sensitive generated content"))
    assert "reach OpenAI" in safe_error(APIConnectionError(request=response.request))
    assert "retry" in safe_error(RuntimeError("secret value"))


def test_missing_openai_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "OPENAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        create_agent("OpenAI", Settings(data_dir=tmp_path))
