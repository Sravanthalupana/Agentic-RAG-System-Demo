---
name: test-app
description: Install, run, and test the Agentic RAG demo; mandatory before every PR.
---

# Test app

Run these steps from the repository root. Read this file before **every** PR.
Bootstrap/ingestion/agent slices do not yet have a UI: run quality gates, and mark UI steps not applicable.

## Install
Use Python 3.11 or 3.12. On a fresh machine:
```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
If your Python 3.11+ binary is named `python`, use `python -m venv .venv` instead.
Never echo a key. For real-provider smoke tests, provision `OPENAI_API_KEY` in the shell environment
(`OPENAL_API_KEY` is accepted as an alias). Defaults: `gpt-4o-mini` and `text-embedding-3-small`.

## Quality gates (all slices)
```sh
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
```
If formatting fails, run `.venv/bin/python -m ruff format .` and rerun all gates.

## Evaluation (once evals/ exists)
```sh
.venv/bin/python -m evals.run --offline
# Paid real-provider comparison; isolated temporary store by default:
.venv/bin/python -m evals.run
```
Offline mode tests the harness with deterministic fake embeddings/LLM, not OpenAI quality.
Confirm both systems are in the table with answer accuracy, citation validity, retrieval count,
and latency. Do not present offline scores as real-provider results.

## Run UI (once app.py exists)
```sh
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Use a temporary store for recording to avoid altering real uploads:
```sh
RAG_DATA_DIR=/tmp/rag-ui-test .venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Recorded browser checks (mandatory for every UI PR)
Use the UI testing agent when available; it owns server setup and recording. Otherwise record
the desktop while exercising these steps in a browser at port 8501:
1. Verify app loads, provider mode is explicit, empty corpus has helpful instructions.
2. Upload `evals/corpus/handbook.txt` and click **Index documents**. Verify chunk count and success.
3. Re-index the same file; verify the chunk count does not grow.
4. Ask "What is the reimbursement deadline?". Verify "30 days", an expandable source with
   filename/location/chunk text, retrieval trace, and latency. Ask a follow-up about the same topic.
5. Ask a greeting ("Hello"); verify no retrieval and no invented citations.
6. Ask "Who is the CEO?"; verify the agent says the documents do not contain the answer.
7. Run `.venv/bin/python -m tests.make_ui_fixtures`. Upload `artifacts/ui-fixtures/warranty.pdf`
   and `support.docx`; query warranty and support facts, verify page/block source locations.
8. Try invalid content in a supported extension; verify a safe error and the app still works.
9. Reload and verify persistence; use **Clear conversation** and verify chat clears but documents remain.
10. Stop recording; attach the video to the UI PR/deliverable, never commit media.

With no API key, use the sidebar's explicitly labeled **Offline demo** mode for UI checks.
That validates wiring only. Record the missing real-provider check as a gap, not a pass.
If a key is available, repeat the primary upload/question/follow-up with OpenAI mode.
Do not record, display, or commit secrets. Report the actual commands, test counts, mode and gaps.
