from __future__ import annotations

from fastapi.testclient import TestClient

from backend import main


class FakeRetriever:
    def search(self, query, top_k):
        return type("Result", (), {
            "final_chunks": [{"source": "test.md", "section": "Тест", "version": "1.0", "text": "Подтверждённый фрагмент"}],
            "sources": [{"source": "test.md", "section": "Тест", "version": "1.0"}],
            "diagnostics": {"vector_results": [{"score": 0.9}], "bm25_results": [{"score": 1.2}], "final_chunks": [], "rrf_k": 60},
        })()


def test_health():
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_returns_sources_and_diagnostics(monkeypatch):
    monkeypatch.setattr(main, "get_retriever", lambda: FakeRetriever())

    async def fake_answer(question, history, chunks):
        assert chunks[0]["source"] == "test.md"
        return "Ответ по документации\nИсточники: 1"

    monkeypatch.setattr(main, "generate_answer", fake_answer)
    response = TestClient(main.app).post("/api/chat", json={"message": "Вопрос"})
    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "test.md"
    assert "bm25_results" in response.json()["diagnostics"]


def test_chat_does_not_call_llm_when_index_is_missing(monkeypatch):
    def missing():
        raise main.HTTPException(503, "Индекс базы знаний не создан")

    monkeypatch.setattr(main, "get_retriever", missing)
    response = TestClient(main.app).post("/api/chat", json={"message": "Вопрос"})
    assert response.status_code == 503


def test_chat_returns_json_for_llm_error(monkeypatch):
    monkeypatch.setattr(main, "get_retriever", lambda: FakeRetriever())

    async def failing_answer(question, history, chunks):
        raise main.LLMError("timeout", "LLM не успел ответить. Повторите запрос.", 504)

    monkeypatch.setattr(main, "generate_answer", failing_answer)
    response = TestClient(main.app).post("/api/chat", json={"message": "Вопрос"})
    assert response.status_code == 504
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == "LLM не успел ответить. Повторите запрос."
