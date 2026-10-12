from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag.config import Settings
from rag.ingestion import ingest
from rag.runtime import create_agent


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    for name in ("OPENAI_API_KEY", "OPENAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_empty_demo_loads(workspace: Path) -> None:
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    assert app.selectbox[0].value == "Offline demo"
    assert app.metric[1].value == "0"
    assert any("Upload a document" in info.value for info in app.info)
    assert app.button[0].disabled


def test_missing_key_shows_safe_error(workspace: Path) -> None:
    app = AppTest.from_file("app.py").run()
    app.selectbox[0].select("OpenAI").run()
    assert not app.exception
    assert "Set OPENAI_API_KEY" in app.error[0].value
    assert app.chat_input[0].disabled


def test_chat_sources_history_latency_and_clear(workspace: Path) -> None:
    settings = Settings(data_dir=workspace)
    agent = create_agent("Offline demo", settings)
    ingest(
        "handbook.txt",
        b"The reimbursement deadline is 30 days. "
        b"Submit reimbursement receipts through the expense portal.",
        agent.store,
        agent.embedder,
        settings,
    )
    app = AppTest.from_file("app.py").run()
    app.chat_input[0].set_value("What is the reimbursement deadline?").run()
    assert not app.exception
    assert any("30 days" in text.value for text in app.text)
    assert any("handbook.txt" in expander.label for expander in app.expander)
    assert "ms" in app.metric[2].value
    app.chat_input[0].set_value("How do I submit it?").run()
    assert any("expense portal" in text.value for text in app.text)
    app.chat_input[0].set_value("Hello").run()
    assert app.metric[3].value == "0"
    app.chat_input[0].set_value("Who is the CEO?").run()
    assert any("could not find" in text.value for text in app.text)
    app.button[1].click().run()
    assert not app.chat_message
    assert app.metric[1].value == "1"
