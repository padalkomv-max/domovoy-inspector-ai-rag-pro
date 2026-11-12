from __future__ import annotations

import pytest

from backend.llm import LLMError, extract_completion_text, resolve_provider_config


def test_gemini_provider_uses_google_ai_studio_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    provider, key, base_url, model = resolve_provider_config()
    assert provider == "gemini"
    assert key == "test-gemini-key"
    assert base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert model == "gemini-3-flash-preview"


def test_openai_configuration_remains_supported(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "test-openai-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    assert resolve_provider_config() == ("openai", "test-openai-key", "https://example.test/v1", "test-model")


def test_unknown_provider_fails_closed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(LLMError) as error:
        resolve_provider_config()
    assert error.value.code == "unsupported_provider"


def test_completion_parser_accepts_gemini_multipart_content():
    data = {"choices": [{"message": {"content": [{"type": "text", "text": "Ответ по базе"}]}}]}
    assert extract_completion_text(data) == "Ответ по базе"


def test_completion_parser_rejects_empty_content():
    with pytest.raises(ValueError, match="content_missing"):
        extract_completion_text({"choices": [{"message": {"content": None}}]})
