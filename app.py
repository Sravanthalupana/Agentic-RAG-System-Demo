import os
from time import perf_counter
from typing import cast

import streamlit as st

from rag.agent import AgentResult
from rag.config import Settings
from rag.ingestion import ingest
from rag.runtime import ProviderMode, create_agent, safe_error


def render_result(result: AgentResult) -> None:
    st.text(result.answer)
    st.caption(
        f"{result.latency_ms:,.0f} ms · {result.retrieval_count} retrievals · "
        f"{'Insufficient evidence' if result.abstained else 'Answered'}"
    )
    for i, hit in enumerate(result.citations, 1):
        with st.expander(f"[{i}] {hit.chunk.source} · {hit.chunk.location}"):
            st.caption(f"Chunk {hit.chunk.id} · cosine score {hit.score:.3f}")
            st.code(hit.chunk.text, language=None, wrap_lines=True)
    with st.expander("Retrieval trace"):
        for step in result.trace:
            st.text(f"{step.action.upper()} · {step.detail}")
            if step.chunk_ids:
                st.caption("Chunks: " + ", ".join(step.chunk_ids))


def main() -> None:
    st.set_page_config(page_title="Agentic RAG · Document chat", page_icon="📚", layout="wide")
    st.markdown(
        """<style>
    .block-container {max-width: 1100px; padding-top: 2.5rem;}
    [data-testid="stMetric"] {background: #f0f4f8; border-radius: 10px; padding: 14px;}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: #19344d;}
    </style>""",
        unsafe_allow_html=True,
    )
    settings = Settings.from_env()
    has_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAL_API_KEY"))
    with st.sidebar:
        st.title("Document workspace")
        mode = cast(
            ProviderMode,
            st.selectbox("Provider", ["OpenAI", "Offline demo"], index=0 if has_key else 1),
        )
        if mode == "Offline demo":
            st.warning("OFFLINE DEMO: lexical search and extractive answers. Not OpenAI quality.")
        else:
            st.caption(f"{settings.chat_model} · {settings.embedding_model}")
        st.caption("PDF, DOCX, TXT, MD · 20 MB per file")
        uploads = st.file_uploader(
            "Upload documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
        )

    if st.session_state.get("provider_mode") != mode:
        st.session_state.messages = []
        st.session_state.provider_mode = mode
    agent = None
    try:
        agent = create_agent(mode, settings)
    except Exception as error:
        st.error(safe_error(error))

    with st.sidebar:
        if st.button("Index documents", type="primary", disabled=not uploads or agent is None):
            assert agent is not None
            started = perf_counter()
            for upload in uploads:
                try:
                    with st.spinner(f"Indexing {upload.name}…"):
                        count = ingest(
                            upload.name, upload.getvalue(), agent.store, agent.embedder, settings
                        )
                    st.success(f"{upload.name}: {count} chunks indexed")
                except Exception as error:
                    st.error(f"{upload.name}: {safe_error(error)}")
            st.caption(f"Indexing time: {(perf_counter() - started) * 1000:,.0f} ms")
        if agent is not None:
            st.subheader("Indexed sources")
            for source in agent.store.sources():
                st.text(source)
            st.caption("Identical files are deduplicated. Same filename replaces the source.")
        if st.button("Clear conversation"):
            st.session_state.messages = []
        st.caption(
            "Vectors and text stay on this server. OpenAI mode sends text to OpenAI. "
            "For local trusted use only; no authentication or per-user document isolation."
        )

    st.title("Agentic RAG")
    st.caption("Ask your documents. Follow the evidence.")
    if mode == "Offline demo":
        st.info(
            "Offline demo is active. To use real embeddings and an LLM, configure "
            "OPENAI_API_KEY on the server and select OpenAI."
        )

    messages = st.session_state.messages
    results = [message["result"] for message in messages if "result" in message]
    last = results[-1] if results else None
    columns = st.columns(4)
    columns[0].metric("Documents", len(agent.store.sources()) if agent else 0)
    columns[1].metric("Source chunks", agent.store.count if agent else 0)
    columns[2].metric("Last answer", f"{last.latency_ms:,.0f} ms" if last else "—")
    columns[3].metric("Retrievals", str(last.retrieval_count) if last else "—")
    if agent and not agent.store.count:
        st.info("Upload a document in the sidebar, then click Index documents to get started.")
    for message in messages:
        with st.chat_message(message["role"]):
            if "result" in message:
                render_result(message["result"])
            else:
                st.text(message["content"])

    if question := st.chat_input(
        "Ask a question about your documents…", max_chars=2000, disabled=agent is None
    ):
        assert agent is not None
        history = [{"role": message["role"], "content": message["content"]} for message in messages]
        with st.chat_message("user"):
            st.text(question)
        try:
            with st.spinner("Deciding what to retrieve…"):
                result = agent.ask(question, history)
            messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": result.answer, "result": result},
                ]
            )
            st.rerun()
        except Exception as error:
            st.error(safe_error(error))


if __name__ == "__main__":
    main()
