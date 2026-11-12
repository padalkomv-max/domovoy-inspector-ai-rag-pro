from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from .diagnostics import redact_error
from .prompt import build_messages


@dataclass
class LLMError(Exception):
    code: str
    public_message: str
    http_status: int = 502


def extract_completion_text(data: Any) -> str:
    if not isinstance(data, dict):
        raise ValueError("response_not_object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("choices_missing")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("message_missing")
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        text = "\n".join(parts)
    else:
        text = ""
    if not text.strip():
        if message.get("refusal"):
            raise ValueError("provider_refusal")
        raise ValueError("content_missing")
    if len(text) > 12000:
        raise ValueError("completion_too_large")
    return text.strip()


def resolve_provider_config() -> tuple[str, str, str, str]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        base_url = os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
        ).rstrip("/")
        model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()
        return provider, key, base_url, model
    if provider == "openai":
        key = os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip()
        return provider, key, base_url, model
    raise LLMError("unsupported_provider", f"Неизвестный LLM_PROVIDER: {provider}", 503)


async def generate_answer(question: str, history: list[dict[str, str]], chunks: list[dict]) -> str:
    provider, key, base_url, model = resolve_provider_config()
    if not key:
        key_name = "GEMINI_API_KEY" if provider == "gemini" else "LLM_API_KEY"
        raise LLMError("missing_api_key", f"{key_name} не настроен. Добавьте ключ в .env и перезапустите сервер.", 503)
    payload = {"model": model, "messages": build_messages(question, history, chunks), "temperature": 0.1}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10), follow_redirects=False) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        content = extract_completion_text(data)
        if key in content:
            raise ValueError("secret_in_completion")
        return content.strip()
    except httpx.TimeoutException as exc:
        raise LLMError("timeout", "LLM не успел ответить. Повторите запрос.", 504) from exc
    except httpx.HTTPStatusError as exc:
        raise LLMError("provider_http_error", "LLM-провайдер вернул ошибку. Проверьте настройки API.", 502) from exc
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        _ = redact_error(str(exc))
        raise LLMError("invalid_provider_response", "Не удалось обработать ответ LLM-провайдера.", 502) from exc
