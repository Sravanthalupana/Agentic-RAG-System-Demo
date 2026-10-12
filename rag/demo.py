"""Explicit offline wiring demo; not a substitute for OpenAI evaluation."""

import hashlib
import re
from collections import Counter
from typing import cast

from rag.llm import Decision

STOP_WORDS = set(
    "a an the what which how is are was were to of for in on and or i it that "
    "do does can me my we our you your please tell about much".split()
)


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - STOP_WORDS


class DemoEmbedder:
    identity = "offline:hashed-words-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * 256
            for word, count in Counter(terms(text)).items():
                bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 256
                vector[bucket] += count
            if not any(vector):
                vector[0] = 1.0
            vectors.append(vector)
        return vectors


class DemoChat:
    def complete(self, system: str, payload: dict[str, object]) -> Decision:
        question = str(payload["question"])
        if question.strip().casefold().rstrip("!.") in {"hi", "hello", "thanks", "thank you"}:
            return Decision(
                action="answer",
                answer="Hello! Ask me about your uploaded documents.",
                document_question=False,
            )
        history = cast(list[dict[str, str]], payload["history"])
        query = question
        if re.search(r"\b(it|that|they|those|their)\b", question.casefold()):
            previous = next(
                (turn["content"] for turn in reversed(history) if turn["role"] == "user"), ""
            )
            query = f"{previous} {question}".strip()[:2000]
        sources = cast(list[dict[str, str]], payload["sources"])
        if not sources and int(cast(int, payload["retrievals_left"])) > 0:
            return Decision(action="retrieve", query=query, reason="Search document terms.")
        keywords = terms(query)
        matches: list[tuple[int, str, str]] = []
        for source in sources:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", source["text"]):
                score = len(keywords & terms(sentence))
                if score:
                    matches.append((score, sentence.strip(), source["id"]))
        if not matches:
            return Decision(
                action="answer",
                answer="I could not find that in the uploaded documents.",
                abstain=True,
            )
        best: list[tuple[int, str, str]] = []
        uncovered = set(keywords)
        while matches and len(best) < 3:
            matches.sort(key=lambda match: (-len(uncovered & terms(match[1])), -match[0]))
            candidate = matches.pop(0)
            if not (uncovered & terms(candidate[1])):
                break
            best.append(candidate)
            uncovered -= terms(candidate[1])
        return Decision(
            action="answer",
            answer=" ".join(dict.fromkeys(item[1] for item in best)),
            citations=list(dict.fromkeys(item[2] for item in best)),
        )
