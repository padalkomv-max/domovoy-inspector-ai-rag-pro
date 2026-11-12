from __future__ import annotations

import json
from pathlib import Path

from backend.rag import parse_markdown, tokenize


ROOT = Path(__file__).resolve().parents[1]


def test_source_bundle_contains_ten_documents_and_index_readme():
    files = list((ROOT / "data" / "raw").glob("*.md"))
    assert len(files) == 11
    assert sum(file.name != "README_индекс.md" for file in files) == 10


def test_chunk_metadata_is_preserved():
    chunks = parse_markdown(ROOT / "data" / "raw" / "09_ошибки_и_диагностика.md")
    assert chunks
    assert all(chunk["source"] == "09_ошибки_и_диагностика.md" for chunk in chunks)
    assert all(chunk["section"] for chunk in chunks)
    assert all(chunk["version"] == "1.0" for chunk in chunks)
    assert all("Источник:" not in chunk["text"] for chunk in chunks)
    assert all(chunk["text"].startswith("Документ:") for chunk in chunks)


def test_tokenizer_keeps_error_codes_and_cyrillic_terms():
    tokens = tokenize("HTTP 400 /api/analyze GEMINI_API_KEY")
    assert "400" in tokens
    assert "api/analyze" in tokens
    assert "gemini_api_key" in tokens


def test_scenarios_are_exactly_ten_and_cover_requested_kinds():
    scenarios = json.loads((ROOT / "tests" / "scenarios.json").read_text(encoding="utf-8"))
    assert len(scenarios) == 10
    assert {item["kind"] for item in scenarios} >= {"direct", "paraphrase", "error", "condition", "exception", "absence", "out_of_scope"}


def test_real_hybrid_index_runs_both_searchers_and_rrf():
    index_dir = ROOT / "data" / "index"
    if not (index_dir / "manifest.json").exists():
        import pytest

        pytest.skip("Индекс ещё не создан")
    from backend.rag import HybridRetriever
    from backend.config import settings

    result = HybridRetriever.load(settings.index_dir).search("Почему ссылка на фото может дать ошибку 400?", top_k=6)
    assert result.diagnostics["vector_results"]
    assert result.diagnostics["bm25_results"]
    assert result.diagnostics["final_chunks"]
    assert any(source["source"] == "09_ошибки_и_диагностика.md" for source in result.sources)


def test_onnx_manifest_dimension_matches_faiss():
    manifest = json.loads((ROOT / "data" / "index" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedding_backend"] == "onnxruntime"
    assert manifest["embedding_dimension"] == 1024
    assert manifest["embedding_max_length"] == 512


def test_procedural_query_expands_the_leading_document_sections():
    index_dir = ROOT / "data" / "index"
    if not (index_dir / "manifest.json").exists():
        import pytest

        pytest.skip("Индекс ещё не создан")
    from backend.rag import HybridRetriever
    from backend.config import settings

    result = HybridRetriever.load(settings.index_dir).search(
        "Как создать новое обращение в приложении Домовой инспектор? Опиши порядок действий", top_k=6
    )
    sections = {chunk["section"] for chunk in result.final_chunks if chunk["source"] == "02_создание_обращения.md"}
    assert {"Начало", "Заполнение", "Отправка на анализ"}.issubset(sections)
