import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import TypedDict

from rag.agent import AgentResult, RAGAgent
from rag.config import Settings
from rag.demo import DemoChat, DemoEmbedder
from rag.embeddings import Embedder, OpenAIEmbedder
from rag.ingestion import ingest
from rag.llm import ChatModel, OpenAIChat
from rag.store import VectorStore

ROOT = Path(__file__).parent


class Case(TypedDict, total=False):
    id: str
    question: str
    history: list[dict[str, str]]
    keywords: list[list[str]]
    sources: list[str]
    abstain: bool


@dataclass(frozen=True)
class Score:
    system: str
    case_id: str
    accuracy: float
    citation_validity: float
    source_coverage: float
    latency_ms: float
    retrievals: int
    answer: str


def score(system: str, case: Case, result: AgentResult) -> Score:
    expected_abstention = case.get("abstain", False)
    matches = all(
        any(word.casefold() in result.answer.casefold() for word in alternatives)
        for alternatives in case["keywords"]
    )
    accuracy = float(result.abstained == expected_abstention and matches)
    retrieved_ids = {
        chunk_id
        for step in result.trace
        if step.action == "retrieve"
        for chunk_id in step.chunk_ids
    }
    valid = sum(hit.chunk.id in retrieved_ids for hit in result.citations)
    validity = valid / len(result.citations) if result.citations else float(not case["sources"])
    actual_sources = {hit.chunk.source for hit in result.citations}
    coverage = (
        len(actual_sources & set(case["sources"])) / len(case["sources"])
        if case["sources"]
        else float(not actual_sources)
    )
    return Score(
        system,
        case["id"],
        accuracy,
        validity,
        coverage,
        result.latency_ms,
        result.retrieval_count,
        result.answer,
    )


def evaluate(agent: RAGAgent, cases: list[Case]) -> list[Score]:
    rows = []
    for i, case in enumerate(cases):
        systems = [("agentic", agent.ask), ("baseline", agent.baseline)]
        for name, run in systems[:: 1 if i % 2 == 0 else -1]:
            result = run(case["question"], case.get("history", []))
            rows.append(score(name, case, result))
    return rows


def comparison_table(rows: list[Score]) -> str:
    lines = [
        "System    | Accuracy | Citation validity | Source coverage | Latency ms | Retrievals",
        "----------|----------|-------------------|-----------------|------------|-----------",
    ]
    for name in ("agentic", "baseline"):
        group = [row for row in rows if row.system == name]
        lines.append(
            f"{name:<9} | {mean(row.accuracy for row in group):>7.1%}  | "
            f"{mean(row.citation_validity for row in group):>16.1%}  | "
            f"{mean(row.source_coverage for row in group):>14.1%}  | "
            f"{mean(row.latency_ms for row in group):>10.1f} | "
            f"{mean(row.retrievals for row in group):>10.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare agentic and single-shot RAG.")
    parser.add_argument("--offline", action="store_true", help="Fake providers; not LLM quality.")
    parser.add_argument("--output", type=Path, help="Optional per-question JSON results.")
    args = parser.parse_args()
    settings = Settings.from_env()
    embedder: Embedder = DemoEmbedder() if args.offline else OpenAIEmbedder(settings)
    model: ChatModel = DemoChat() if args.offline else OpenAIChat(settings)
    cases: list[Case] = json.loads((ROOT / "questions.json").read_text())
    with TemporaryDirectory(prefix="rag-eval-") as directory:
        store = VectorStore(Path(directory), embedder.identity)
        for path in sorted((ROOT / "corpus").glob("*.txt")):
            ingest(path.name, path.read_bytes(), store, embedder, settings)
        rows = evaluate(RAGAgent(store, embedder, model, settings), cases)
    mode = "OFFLINE DEMO — wiring only, not OpenAI quality" if args.offline else "OpenAI"
    print(f"Mode: {mode} | {len(cases)} fixed questions | same corpus/models/top_k")
    print(comparison_table(rows))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"mode": mode, "results": [asdict(row) for row in rows]}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
