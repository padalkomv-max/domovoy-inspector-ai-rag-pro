from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app


pytestmark = pytest.mark.skipif(not os.getenv("RUN_LIVE_LLM") or not os.getenv("LLM_API_KEY"), reason="Нужен явный RUN_LIVE_LLM=1 и LLM_API_KEY")


@pytest.mark.parametrize("question", [
    "Как создать новое обращение?",
    "Можно ли анализировать проблему без фото?",
    "Почему URL фотографии даёт HTTP 400?",
    "Сохраняется ли история обращений?",
    "Есть ли отправка обращения в УК автоматически?",
])
def test_live_answer_is_not_empty_and_contains_source_marker(question):
    response = TestClient(app).post("/api/chat", json={"message": question})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"].strip()
    assert "sources" in data
    assert "diagnostics" in data
