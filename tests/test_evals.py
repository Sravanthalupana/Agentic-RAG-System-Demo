import json
from pathlib import Path

from evals.run import ROOT, Case, comparison_table, evaluate, score
from rag.agent import AgentResult, RAGAgent, TraceStep
from rag.config import Settings
from rag.demo import DemoChat, DemoEmbedder
from rag.ingestion import ingest
from rag.models import Chunk, Hit
from rag.store import VectorStore


def test_offline_comparison_is_repeatable(tmp_path: Path) -> None:
    embedder = DemoEmbedder()
    store = VectorStore(tmp_path, embedder.identity)
    for path in (ROOT / "corpus").glob("*.txt"):
        ingest(path.name, path.read_bytes(), store, embedder, Settings())
    cases: list[Case] = json.loads((ROOT / "questions.json").read_text())
    agent = RAGAgent(store, embedder, DemoChat())
    first, second = evaluate(agent, cases), evaluate(agent, cases)
    assert len(first) == 16
    assert [(row.system, row.accuracy, row.answer) for row in first] == [
        (row.system, row.accuracy, row.answer) for row in second
    ]
    assert all(row.retrievals == 1 for row in first if row.system == "baseline")
    assert any(row.retrievals == 0 for row in first if row.system == "agentic")
    assert "baseline" in comparison_table(first) and "agentic" in comparison_table(first)
    assert "Citation validity" in comparison_table(first)


def test_scoring_detects_wrong_answer_and_invalid_citation() -> None:
    case: Case = {
        "id": "fact",
        "question": "Deadline?",
        "keywords": [["30 days"]],
        "sources": ["handbook.txt"],
    }
    wrong = AgentResult(
        "99 days",
        [Hit(Chunk("fake", "other.txt", "text", ""), 1.0)],
        [TraceStep("retrieve", "deadline", ["real"])],
        10.0,
        False,
    )
    result = score("agentic", case, wrong)
    assert result.accuracy == result.citation_validity == result.source_coverage == 0


def test_scoring_abstention() -> None:
    case: Case = {
        "id": "unknown",
        "question": "CEO?",
        "keywords": [],
        "sources": [],
        "abstain": True,
    }
    result = score("agentic", case, AgentResult("Unknown", [], [], 0, True))
    assert result.accuracy == result.citation_validity == 1
