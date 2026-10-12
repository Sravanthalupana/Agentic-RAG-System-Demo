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
## Document ingestion

With your key in the environment:

```sh
.venv/bin/python -m rag.ingestion path/to/handbook.pdf path/to/notes.docx path/to/policy.txt
```

The loader keeps PDF page and DOCX block/table locations, uses overlapping token-bounded chunks,
embeds batches through OpenAI, and persists vectors and source text transactionally in SQLite.
Local FAISS cosine search ranks those vectors. Re-uploading identical content skips embedding;
changed content replaces that filename's chunks. Embedding model/dimension mismatches are rejected.
Files with the same basename represent the same source. Limits: 20 MB/file, 500 PDF pages,
2 million extracted characters, UTF-8 text. Scanned PDFs need external OCR.

Agent evaluation and the chat UI arrive in subsequent slices.
