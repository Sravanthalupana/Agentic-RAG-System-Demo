# Agentic RAG System Demo

A local-first document question-answering demo using OpenAI, FAISS, and Streamlit.

## Setup

Python 3.11+ (3.12 recommended):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Provide `OPENAI_API_KEY` through your environment or secret manager, never in code.
`OPENAL_API_KEY` is also accepted. Optional `RAG_CHAT_MODEL`, `RAG_EMBEDDING_MODEL`, and
`RAG_DATA_DIR` override model names and the local `.data/` storage directory.

## Development

Read `AGENTS.md` and use `.agents/skills/test-app/SKILL.md` before every PR.

```sh
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
```

Tests use fake providers and do not require an API key. GitHub Actions runs the same gates.
PDF/DOCX/text ingestion, agent evaluation, and the chat UI arrive in subsequent slices.
