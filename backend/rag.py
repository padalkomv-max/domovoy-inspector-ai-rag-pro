from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

TOKEN_RE = re.compile(r"[\w]+(?:[-/][\w]+)*", re.UNICODE)
RRF_K = 60


class IndexNotReadyError(RuntimeError):
    pass


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("ё", "е"))


def is_procedural_query(query: str) -> bool:
    tokens = tokenize(query)
    return "как" in tokens and any(
        token.startswith(("созда", "поряд", "действ", "шаг", "инструк", "процед")) for token in tokens
    )


def parse_markdown(path: Path, chunk_size: int = 1100, overlap: int = 140) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    version_match = re.search(r"^Версия документа:\s*(.+)$", content, re.MULTILINE)
    version = version_match.group(1).strip() if version_match else "не указана"
    lines = content.splitlines()
    title = path.stem
    chunks: list[dict[str, Any]] = []
    current_section = title
    section_lines: list[str] = []
    seen_section = False

    def flush() -> None:
        nonlocal section_lines
        text = "\n".join(section_lines).strip()
        if not text:
            section_lines = []
            return
        pieces = [text] if len(text) <= chunk_size else [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        for piece_index, piece in enumerate(pieces):
            retrieval_text = f"Документ: {title}\nРаздел: {current_section}\n{piece.strip()}"
            chunks.append(
                {
                    "id": hashlib.sha1(f"{path.name}|{current_section}|{piece_index}|{retrieval_text}".encode("utf-8")).hexdigest()[:16],
                    "source": path.name,
                    "section": current_section,
                    "version": version,
                    "text": retrieval_text,
                }
            )
        section_lines = []

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            current_section = title
        elif line.startswith("## "):
            if not seen_section:
                section_lines = []
                seen_section = True
            flush()
            current_section = line[3:].strip()
        else:
            section_lines.append(line)
    flush()
    return chunks


def _load_dependencies():
    try:
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Не установлены зависимости RAG. Выполните: python -m pip install -r requirements.txt"
        ) from exc
    return faiss, BM25Okapi, SentenceTransformer


def build_index(source_dir: Path, index_dir: Path, embedding_model: str) -> dict[str, Any]:
    faiss, BM25Okapi, SentenceTransformer = _load_dependencies()
    source_files = sorted(p for p in source_dir.glob("*.md") if p.name != "README_индекс.md")
    if len(source_files) != 10:
        raise RuntimeError(f"Ожидалось 10 Markdown-документов, найдено: {len(source_files)}")
    chunks = [chunk for path in source_files for chunk in parse_markdown(path)]
    model = SentenceTransformer(embedding_model)
    vectors = model.encode([chunk["text"] for chunk in chunks], normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "vectors.faiss").write_bytes(faiss.serialize_index(index).tobytes())
    tokenized = [tokenize(chunk["text"] + " " + chunk["section"] + " " + chunk["source"]) for chunk in chunks]
    BM25Okapi(tokenized)
    (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (index_dir / "bm25.json").write_text(json.dumps({"corpus": tokenized, "k1": 1.5, "b": 0.75}, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(source_files),
        "chunks": len(chunks),
        "embedding_model": embedding_model,
        "vector_index": "FAISS IndexFlatIP with normalized embeddings",
        "keyword_index": "rank_bm25 BM25Okapi",
        "sources": [path.name for path in source_files],
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "chunks": len(chunks), "documents": len(source_files), "model": embedding_model}


@dataclass
class SearchResult:
    final_chunks: list[dict[str, Any]]
    sources: list[dict[str, str]]
    diagnostics: dict[str, Any]


class HybridRetriever:
    def __init__(self, index: Any, chunks: list[dict[str, Any]], bm25: Any, model: Any) -> None:
        self.index = index
        self.chunks = chunks
        self.bm25 = bm25
        self.model = model

    @classmethod
    def load(cls, index_dir: Path) -> "HybridRetriever":
        required = [index_dir / "vectors.faiss", index_dir / "chunks.json", index_dir / "bm25.json", index_dir / "manifest.json"]
        if not all(path.exists() for path in required):
            raise IndexNotReadyError("Индекс базы знаний не создан. Выполните: python scripts/reindex.py")
        faiss, BM25Okapi, SentenceTransformer = _load_dependencies()
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        bm25_data = json.loads((index_dir / "bm25.json").read_text(encoding="utf-8"))
        return cls(
            faiss.deserialize_index(np.frombuffer((index_dir / "vectors.faiss").read_bytes(), dtype=np.uint8)),
            chunks,
            BM25Okapi(bm25_data["corpus"], k1=bm25_data["k1"], b=bm25_data["b"]),
            SentenceTransformer(manifest["embedding_model"]),
        )

    def search(self, query: str, top_k: int = 6) -> SearchResult:
        vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        distances, vector_ids = self.index.search(vector, min(max(top_k * 2, 12), len(self.chunks)))
        vector_results = []
        for rank, (idx, score) in enumerate(zip(vector_ids[0], distances[0]), start=1):
            if idx < 0:
                continue
            vector_results.append({"chunk_id": self.chunks[int(idx)]["id"], "rank": rank, "score": round(float(score), 4), "source": self.chunks[int(idx)]["source"], "section": self.chunks[int(idx)]["section"]})
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ids = sorted(range(len(bm25_scores)), key=lambda i: float(bm25_scores[i]), reverse=True)[: min(max(top_k * 2, 12), len(self.chunks))]
        bm25_results = [{"chunk_id": self.chunks[idx]["id"], "rank": rank, "score": round(float(bm25_scores[idx]), 4), "source": self.chunks[idx]["source"], "section": self.chunks[idx]["section"]} for rank, idx in enumerate(bm25_ids, start=1)]
        by_id = {chunk["id"]: chunk for chunk in self.chunks}
        rrf_scores: dict[str, float] = {}
        for result in vector_results:
            rrf_scores[result["chunk_id"]] = rrf_scores.get(result["chunk_id"], 0) + 1 / (RRF_K + result["rank"])
        for result in bm25_results:
            rrf_scores[result["chunk_id"]] = rrf_scores.get(result["chunk_id"], 0) + 1 / (RRF_K + result["rank"])
        selected_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        if is_procedural_query(query) and bm25_ids:
            lead_source = self.chunks[bm25_ids[0]]["source"]
            source_chunks = [chunk["id"] for chunk in self.chunks if chunk["source"] == lead_source]
            expanded_ids = source_chunks[:top_k]
            selected_ids = expanded_ids + [chunk_id for chunk_id in selected_ids if chunk_id not in expanded_ids]
            selected_ids = selected_ids[:top_k]
        final_chunks = []
        for chunk_id in selected_ids:
            chunk = dict(by_id[chunk_id])
            chunk["rrf_score"] = round(rrf_scores[chunk_id], 6) if chunk_id in rrf_scores else None
            chunk["selection"] = "rrf" if chunk_id in rrf_scores else "source_expansion"
            final_chunks.append(chunk)
        sources = []
        seen = set()
        for chunk in final_chunks:
            key = (chunk["source"], chunk["section"], chunk["version"])
            if key not in seen:
                sources.append({"source": key[0], "section": key[1], "version": key[2]})
                seen.add(key)
        diagnostics = {"vector_results": vector_results, "bm25_results": bm25_results, "final_chunks": final_chunks, "rrf_k": RRF_K}
        return SearchResult(final_chunks, sources, diagnostics)
