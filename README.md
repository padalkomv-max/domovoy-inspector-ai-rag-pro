# Домовой инспектор AI — Оператор техподдержки RAG PRO

Рабочее локальное веб-приложение: русскоязычный AI-чат, который отвечает по базе знаний «Домовой инспектор AI» через настоящий гибридный RAG-поиск.

## Что реализовано

- 10 реальных Markdown-документов из архива Google Drive; `README_индекс.md` не индексируется.
- Чанкинг по разделам `##` с сохранением `source`, `section` и `version`.
- Реальные embeddings через `sentence-transformers` и векторный индекс FAISS.
- Настоящий BM25 через `rank-bm25` по тексту, разделу и имени источника.
- RRF объединяет оба ранжирования, удаляет дубли и передаёт LLM только top-K найденных фрагментов.
- Источники отображаются рядом с ответом; «Диагностика поиска» раскрывает vector search, BM25 и итоговый контекст LLM.
- API-ключ нужен только серверу через `.env`; во frontend ключа нет.
- Переиндексация без изменения кода: `python scripts/reindex.py` или защищённый `POST /api/admin/reindex`.

## Структура

```text
backend/
  main.py       FastAPI и API маршруты
  rag.py        чанкинг, embeddings, FAISS, BM25, RRF
  llm.py        OpenAI-совместимый Chat Completions API
  prompt.py     строгие правила ответа по найденному контексту
frontend/       русскоязычный чат без API-ключей
data/raw/       исходные 10 Markdown-документов из архива
data/index/     генерируемые FAISS/BM25 индексы и манифест
scripts/reindex.py
tests/          тесты индексации, поиска, API и frontend-сценарии
```

## Запуск в Windows PowerShell

```powershell
cd "C:\Users\User\Documents\ChatGPT\Кодекс_1"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

### Gemini через Google AI Studio

Для бесплатного Gemini API укажите в `.env`:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=ваш_ключ_из_Google_AI_Studio
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL=gemini-3-flash-preview
```

Ключ не публикуйте и не добавляйте во frontend. Совместимый endpoint сохраняет текущий RAG-контекст: в Gemini отправляются только top-K фрагменты после FAISS + BM25 + RRF.

Для прежней OpenAI-конфигурации оставьте `LLM_PROVIDER=openai` и используйте `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.

Сначала создайте индекс. При первом запуске `sentence-transformers` скачает модель embeddings:

```powershell
.\.venv\Scripts\python.exe scripts\reindex.py
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Откройте http://127.0.0.1:8000. Проверка сервера: http://127.0.0.1:8000/health.

Если документы обновились, замените файлы в `data/raw/` и снова выполните `scripts\reindex.py`. Код менять не требуется. Ожидается ровно 10 Markdown-документов; индексный README исключён намеренно.

## API

- `GET /api/knowledge/status` — статус индекса.
- `POST /api/chat` — `{ "message": "...", "history": [...] }`.
- `POST /api/admin/reindex` — переиндексация при заданном `REINDEX_TOKEN` и заголовке `X-Reindex-Token`.

Без ключа выбранного провайдера приложение не имитирует ответ: поиск можно проверить через тесты и `/api/knowledge/status`, а чат вернёт понятную ошибку настройки.

## Тесты

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/test_frontend.cjs
```

Тестовый набор содержит 10 вопросов: прямые, перефразированные, по ошибкам, условиям и исключениям, а также вопросы, для которых в документации нет ответа. Live-проверка LLM выполняется только при явном наличии ключа:

```powershell
$env:RUN_LIVE_LLM="1"
.\.venv\Scripts\python.exe -m pytest tests/test_live.py -v
Remove-Item Env:RUN_LIVE_LLM
```

Live-тесты не заявляются пройденными без реального ключа и фактических ответов провайдера.
