from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .llm import LLMError, generate_answer
from .rag import HybridRetriever, IndexNotReadyError, build_index

load_dotenv(settings.root / ".env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("domovoy-rag")

app = FastAPI(title="Домовой инспектор AI — Оператор техподдержки RAG PRO", version="1.0.0")
app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")
_retriever: HybridRetriever | None = None


class HistoryItem(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class ReindexResponse(BaseModel):
    status: str
    chunks: int
    documents: int
    model: str


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is not None:
        return _retriever
    try:
        _retriever = HybridRetriever.load(
            settings.index_dir,
            settings.embedding_cache_dir,
            settings.embedding_max_length,
        )
        return _retriever
    except IndexNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/knowledge/status")
async def knowledge_status() -> dict[str, Any]:
    manifest_path = settings.index_dir / "manifest.json"
    if not manifest_path.exists():
        return {"ready": False, "message": "Индекс не создан. Запустите scripts/reindex.py."}
    return {"ready": True, **json.loads(manifest_path.read_text(encoding="utf-8"))}


@app.post("/api/admin/reindex", response_model=ReindexResponse)
async def reindex(x_reindex_token: str | None = Header(default=None)) -> ReindexResponse:
    global _retriever
    if not settings.reindex_token or x_reindex_token != settings.reindex_token:
        raise HTTPException(status_code=403, detail="Недостаточно прав для переиндексации")
    result = build_index(
        settings.source_dir,
        settings.index_dir,
        settings.embedding_model,
        settings.embedding_revision,
        settings.embedding_cache_dir,
        settings.embedding_max_length,
    )
    _retriever = None
    return ReindexResponse(**result)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    retriever = get_retriever()
    result = retriever.search(body.message, top_k=settings.top_k)
    messages = [item.model_dump() for item in body.history]
    try:
        answer = await generate_answer(body.message, messages, result.final_chunks)
    except LLMError as exc:
        logger.warning("LLM request failed: %s", exc.code)
        raise HTTPException(status_code=exc.http_status, detail=exc.public_message) from None
    return ChatResponse(reply=answer, sources=result.sources, diagnostics=result.diagnostics)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
