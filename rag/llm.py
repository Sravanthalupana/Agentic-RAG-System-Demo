import json
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from rag.config import Settings, api_key


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["retrieve", "answer"]
    query: str = Field(default="", max_length=2000)
    answer: str = Field(default="", max_length=8000)
    citations: list[str] = Field(default_factory=list, max_length=12)
    abstain: bool = False
    document_question: bool = True
    reason: str = Field(default="", max_length=500)


class ChatModel(Protocol):
    def complete(self, system: str, payload: dict[str, object]) -> Decision: ...


class OpenAIChat:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.chat_model
        self.client = OpenAI(api_key=api_key(), timeout=45.0, max_retries=2)

    def complete(self, system: str, payload: dict[str, object]) -> Decision:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1800,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned an empty response.")
        return Decision.model_validate_json(content)
