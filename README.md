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

## Retrieval agent

```sh
.venv/bin/python -m rag.agent "What is the reimbursement deadline?"
```

The LLM decides whether to retrieve, rewrites queries with conversation history, and can search
up to three times before answering. Small talk can skip retrieval. Document answers require
citation IDs present in retrieved chunks; invalid/missing IDs trigger abstention. Retrieved content
is marked untrusted. These guards validate citation provenance, not semantic truth or immunity to
prompt injection. Sources include filename, page/block, chunk ID, and quoted text.

## Evaluation

```sh
.venv/bin/python -m evals.run --offline
.venv/bin/python -m evals.run --output artifacts/eval-openai.json
```

Eight fixed questions cover facts, follow-ups, multiple sources, a greeting, and an unanswerable
question. Both systems share the corpus, embeddings, answer model/prompt, history, top-k, and
citation guards. The **baseline** embeds the raw question once and makes one answer call;
the **agentic** system chooses zero to three retrievals and may reformulate queries.

The table reports keyword/abstention **accuracy**, retrieved-ID **citation validity**, expected-source
**coverage**, average end-to-end answer **latency**, and average **retrieval count**. Latency excludes
corpus ingestion; question order alternates which system runs first. Scores are transparent smoke
metrics, not a semantic judge or statistically meaningful benchmark. Missing citations score zero
on document questions; no-citation greetings/abstentions score valid. Per-question JSON retains answers.
Evaluation uses a fresh temporary store, never your uploaded corpus. OpenAI mode makes paid calls.

`--offline` uses hashed lexical embeddings and extractive fake answers to validate wiring; **its
scores do not represent OpenAI performance**.

## Web UI

```sh
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501` on your own machine. Select **OpenAI** (needs a server-side key)
or the clearly labeled **Offline demo**, upload documents, and click **Index documents**.
Ask questions and follow-ups; expand sources to see the cited text and provenance, or the retrieval
trace to see actual search queries. Each response reports latency and retrieval count.
**Clear conversation** clears chat history but keeps indexed documents. Reloads retain the store,
not the chat. Provider switches clear chat and isolate offline vectors in `.data/offline/`.

### Privacy and limits

This is a trusted, single-workspace **local demo**, not a production multi-tenant service. Anyone
with access to the server can query its shared documents. Do not expose it publicly without adding
authentication and per-user storage. Documents are stored unencrypted locally; OpenAI mode sends
document chunks and question/history text to OpenAI. Offline mode sends no content to OpenAI.
No OCR, image extraction, streaming generation, or production authorization is included.

Generate browser-test fixtures with `.venv/bin/python -m tests.make_ui_fixtures`; follow
`.agents/skills/test-app/SKILL.md` for recorded browser tests before UI PRs.
