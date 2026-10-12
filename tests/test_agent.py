from collections import deque
from pathlib import Path

import pytest

from rag.agent import RAGAgent
from rag.config import Settings
from rag.llm import Decision
from rag.models import Chunk
from rag.store import VectorStore
from tests.test_store import FakeEmbedder


class ScriptedChat:
    def __init__(self, decisions: list[Decision]) -> None:
        self.decisions = deque(decisions)
        self.payloads: list[dict[str, object]] = []
        self.systems: list[str] = []

    def complete(self, system: str, payload: dict[str, object]) -> Decision:
        self.payloads.append(payload)
        self.systems.append(system)
        return self.decisions.popleft()


def build_agent(
    tmp_path: Path, decisions: list[Decision], settings: Settings | None = None
) -> RAGAgent:
    embedder = FakeEmbedder()
    store = VectorStore(tmp_path, embedder.identity)
    store.replace([Chunk("a", "handbook.txt", "text", "Deadline: 30 days.")], [[1.0, 0.0]])
    return RAGAgent(store, embedder, ScriptedChat(decisions), settings)


def test_retrieves_and_returns_resolved_citations(tmp_path: Path) -> None:
    agent = build_agent(
        tmp_path,
        [
            Decision(action="retrieve", query="reimbursement deadline"),
            Decision(action="answer", answer="30 days", citations=["a", "a"]),
        ],
    )
    result = agent.ask("When?")
    assert result.answer == "30 days [1]"
    assert result.citations[0].chunk.source == "handbook.txt"
    assert result.retrieval_count == 1 and result.latency_ms >= 0
    assert not result.abstained


def test_skip_retrieval_for_smalltalk(tmp_path: Path) -> None:
    agent = build_agent(
        tmp_path, [Decision(action="answer", answer="Hello", document_question=False)]
    )
    result = agent.ask("Hello")
    assert result.retrieval_count == 0
    assert not result.citations and not result.abstained


@pytest.mark.parametrize(
    "decision",
    [
        Decision(action="answer", answer="30 days", citations=["fake"]),
        Decision(action="answer", answer="30 days"),
        Decision(action="answer", answer="", citations=["a"]),
    ],
)
def test_rejects_fabricated_or_missing_citations(tmp_path: Path, decision: Decision) -> None:
    agent = build_agent(tmp_path, [Decision(action="retrieve", query="deadline"), decision])
    result = agent.ask("Deadline?")
    assert result.abstained and not result.citations
    assert any(step.action == "guard" for step in result.trace)


def test_rejects_document_answer_without_retrieval(tmp_path: Path) -> None:
    result = build_agent(tmp_path, [Decision(action="answer", answer="Invented fact")]).ask(
        "Deadline?"
    )
    assert result.abstained


def test_multiple_queries_and_history(tmp_path: Path) -> None:
    agent = build_agent(
        tmp_path,
        [
            Decision(action="retrieve", query="reimbursement deadline"),
            Decision(action="retrieve", query="expense submission portal"),
            Decision(action="answer", answer="30 days", citations=["a"]),
        ],
    )
    history = [{"role": "user", "content": "Reimbursements?"}]
    result = agent.ask("How do I submit it?", history)
    assert result.retrieval_count == 2
    assert isinstance(agent.model, ScriptedChat)
    assert agent.model.payloads[0]["history"] == history
    assert agent.model.payloads[2]["retrievals_left"] == 1
    assert "UNTRUSTED DATA" in agent.model.systems[0]


@pytest.mark.parametrize("queries", [["same", "SAME"], ["a", "b", "c", "d"], [""]])
def test_bounded_and_deduplicated_queries(tmp_path: Path, queries: list[str]) -> None:
    agent = build_agent(tmp_path, [Decision(action="retrieve", query=query) for query in queries])
    result = agent.ask("test?")
    assert result.abstained and result.retrieval_count <= 3


def test_baseline_exactly_one_retrieval(tmp_path: Path) -> None:
    agent = build_agent(tmp_path, [Decision(action="answer", answer="30 days", citations=["a"])])
    result = agent.baseline("Deadline?")
    assert result.retrieval_count == 1
    assert isinstance(agent.model, ScriptedChat)
    assert len(agent.model.payloads) == 1
    assert agent.model.payloads[0]["retrievals_left"] == 0


def test_empty_corpus_and_invalid_questions(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    chat = ScriptedChat(
        [
            Decision(action="retrieve", query="deadline"),
            Decision(action="answer", answer="Unknown", abstain=True),
        ]
    )
    agent = RAGAgent(VectorStore(tmp_path, embedder.identity), embedder, chat)
    assert agent.ask("Deadline?").abstained
    assert embedder.calls == 0
    for invalid in ("", "   ", "x" * 2001):
        with pytest.raises(ValueError):
            agent.ask(invalid)
        with pytest.raises(ValueError):
            agent.baseline(invalid)
