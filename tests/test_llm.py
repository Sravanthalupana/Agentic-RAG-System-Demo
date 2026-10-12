from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rag.config import Settings
from rag.llm import OpenAIChat


def test_chat_sdk_json_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch("rag.llm.OpenAI") as client:
        create = client.return_value.chat.completions.create
        create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"action":"retrieve","query":"expense policy"}'
                    )
                )
            ]
        )
        model = OpenAIChat(Settings())
        decision = model.complete("Return JSON", {"question": "test"})
        assert decision.query == "expense policy"
        assert create.call_args.kwargs["response_format"] == {"type": "json_object"}
        assert create.call_args.kwargs["messages"][1]["role"] == "user"


@pytest.mark.parametrize("content", [None, "not json", '{"action":"execute_shell"}'])
def test_rejects_invalid_model_response(
    monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch("rag.llm.OpenAI") as client:
        client.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        with pytest.raises(ValueError):
            OpenAIChat(Settings()).complete("JSON", {})
