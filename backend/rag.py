from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
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


STATIC_ONNX_TOKENIZER_FILE = "0_StaticEmbedding/tokenizer.json"
STATIC_ONNX_MODEL_FILE = "onnx/model_quantized.onnx"
DEFAULT_EMBEDDING_REVISION = "b68f4122911bcffcd6e1f695f2d99cd6788972d8"


def _load_dependencies():
    try:
        import faiss
        import httpx
        import onnxruntime as ort
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "Не установлены зависимости RAG. Выполните: python -m pip install -r requirements.txt"
        ) from exc
    return faiss, BM25Okapi, ort, httpx


def _download_model_file(
    model_id: str,
    revision: str,
    relative_path: str,
    cache_dir: str | None,
) -> str:
    _, _, _, httpx = _load_dependencies()
    root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "domovoy-rag-pro"
    target = root / model_id.replace("/", "--") / revision / relative_path
    if target.exists() and target.stat().st_size > 0:
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{relative_path}"
    timeout = httpx.Timeout(None, connect=30)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
    partial.replace(target)
    return str(target)


class BertWordPieceTokenizer:
    """Small dependency-free tokenizer for the model's BERT tokenizer.json."""

    def __init__(self, tokenizer_data: dict[str, Any], max_length: int) -> None:
        self.vocab = tokenizer_data["model"]["vocab"]
        self.unk_id = int(self.vocab[tokenizer_data["model"]["unk_token"]])
        self.cls_id = int(self.vocab["[CLS]"])
        self.sep_id = int(self.vocab["[SEP]"])
        self.pad_id = int(self.vocab["[PAD]"])
        self.max_length = max_length

    @classmethod
    def from_file(cls, path: str, max_length: int) -> "BertWordPieceTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), max_length)

    @staticmethod
    def _clean_text(text: str) -> str:
        output = []
        for char in text:
            codepoint = ord(char)
            if codepoint in (0, 0xFFFD) or unicodedata.category(char) == "Cc":
                continue
            output.append(" " if char.isspace() else char)
        return "".join(output)

    @staticmethod
    def _split_chinese_chars(text: str) -> str:
        output = []
        for char in text:
            codepoint = ord(char)
            if (
                0x4E00 <= codepoint <= 0x9FFF
                or 0x3400 <= codepoint <= 0x4DBF
                or 0x20000 <= codepoint <= 0x2A6DF
                or 0x2A700 <= codepoint <= 0x2B73F
                or 0x2B740 <= codepoint <= 0x2B81F
                or 0x2B820 <= codepoint <= 0x2CEAF
                or 0xF900 <= codepoint <= 0xFAFF
                or 0x2F800 <= codepoint <= 0x2FA1F
            ):
                output.extend((" ", char, " "))
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _strip_accents(text: str) -> str:
        return "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )

    @staticmethod
    def _is_punctuation(char: str) -> bool:
        codepoint = ord(char)
        return (
            33 <= codepoint <= 47
            or 58 <= codepoint <= 64
            or 91 <= codepoint <= 96
            or 123 <= codepoint <= 126
            or unicodedata.category(char).startswith("P")
        )

    @classmethod
    def _basic_tokens(cls, text: str) -> list[str]:
        text = cls._split_chinese_chars(cls._clean_text(text)).lower()
        tokens = []
        for token in text.split():
            token = cls._strip_accents(token)
            start = 0
            for index, char in enumerate(token):
                if cls._is_punctuation(char):
                    if start < index:
                        tokens.append(token[start:index])
                    tokens.append(char)
                    start = index + 1
            if start < len(token):
                tokens.append(token[start:])
        return [token for token in tokens if token]

    def _wordpiece_tokens(self, token: str) -> list[int]:
        if len(token) > 100:
            return [self.unk_id]
        sub_tokens = []
        start = 0
        while start < len(token):
            end = len(token)
            current = None
            while start < end:
                piece = token[start:end]
                if start:
                    piece = "##" + piece
                if piece in self.vocab:
                    current = int(self.vocab[piece])
                    break
                end -= 1
            if current is None:
                return [self.unk_id]
            sub_tokens.append(current)
            start = end
        return sub_tokens

    def encode_batch(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encoded = []
        for text in texts:
            token_ids = [piece for token in self._basic_tokens(text) for piece in self._wordpiece_tokens(token)]
            token_ids = token_ids[: self.max_length - 2]
            encoded.append([self.cls_id, *token_ids, self.sep_id])
        width = max((len(item) for item in encoded), default=2)
        input_ids = np.full((len(encoded), width), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(encoded), width), dtype=np.int64)
        for row, item in enumerate(encoded):
            input_ids[row, : len(item)] = item
            attention_mask[row, : len(item)] = 1
        return input_ids, attention_mask


class OnnxEmbedder:
    """CPU-only embedding runtime without PyTorch or sentence-transformers."""

    def __init__(
        self,
        model_id: str,
        revision: str = DEFAULT_EMBEDDING_REVISION,
        cache_dir: str | None = None,
        max_length: int = 512,
    ) -> None:
        _, _, ort, _ = _load_dependencies()
        tokenizer_path = _download_model_file(model_id, revision, STATIC_ONNX_TOKENIZER_FILE, cache_dir)
        model_path = _download_model_file(model_id, revision, STATIC_ONNX_MODEL_FILE, cache_dir)

        self.tokenizer = BertWordPieceTokenizer.from_file(tokenizer_path, max_length)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(os.getenv("ONNX_INTRA_OP_THREADS", "1"))
        session_options.inter_op_num_threads = int(os.getenv("ONNX_INTER_OP_THREADS", "1"))
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        output_shape = self.session.get_outputs()[0].shape
        self.dimension = int(output_shape[-1])
        self.model_id = model_id
        self.revision = revision

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        input_ids, attention_mask = self.tokenizer.encode_batch(texts)
        output = self.session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        vectors = np.asarray(output, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


def build_index(
    source_dir: Path,
    index_dir: Path,
    embedding_model: str,
    embedding_revision: str = DEFAULT_EMBEDDING_REVISION,
    embedding_cache_dir: str | None = None,
    embedding_max_length: int = 512,
) -> dict[str, Any]:
    faiss, BM25Okapi, _, _ = _load_dependencies()
    source_files = sorted(p for p in source_dir.glob("*.md") if p.name != "README_индекс.md")
    if len(source_files) != 10:
        raise RuntimeError(f"Ожидалось 10 Markdown-документов, найдено: {len(source_files)}")
    chunks = [chunk for path in source_files for chunk in parse_markdown(path)]
    model = OnnxEmbedder(embedding_model, embedding_revision, embedding_cache_dir, embedding_max_length)
    vectors = model.encode([chunk["text"] for chunk in chunks])
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
        "embedding_backend": "onnxruntime",
        "embedding_revision": embedding_revision,
        "embedding_dimension": int(vectors.shape[1]),
        "embedding_max_length": embedding_max_length,
        "vector_index": "FAISS IndexFlatIP with normalized embeddings",
        "keyword_index": "rank_bm25 BM25Okapi",
        "sources": [path.name for path in source_files],
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "chunks": len(chunks),
        "documents": len(source_files),
        "model": embedding_model,
    }


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
    def load(
        cls,
        index_dir: Path,
        embedding_cache_dir: str | None = None,
        embedding_max_length: int = 512,
    ) -> "HybridRetriever":
        required = [index_dir / "vectors.faiss", index_dir / "chunks.json", index_dir / "bm25.json", index_dir / "manifest.json"]
        if not all(path.exists() for path in required):
            raise IndexNotReadyError("Индекс базы знаний не создан. Выполните: python scripts/reindex.py")
        faiss, BM25Okapi, _, _ = _load_dependencies()
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("embedding_backend") != "onnxruntime":
            raise IndexNotReadyError(
                "Индекс создан прежним embedding-runtime. Выполните переиндексацию для ONNX Runtime."
            )
        chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        bm25_data = json.loads((index_dir / "bm25.json").read_text(encoding="utf-8"))
        vector_index = faiss.deserialize_index(
            np.frombuffer((index_dir / "vectors.faiss").read_bytes(), dtype=np.uint8)
        )
        model = OnnxEmbedder(
            manifest["embedding_model"],
            manifest.get("embedding_revision", DEFAULT_EMBEDDING_REVISION),
            embedding_cache_dir,
            int(manifest.get("embedding_max_length", embedding_max_length)),
        )
        expected_dimension = int(manifest.get("embedding_dimension", vector_index.d))
        if vector_index.d != expected_dimension or model.dimension != vector_index.d:
            raise IndexNotReadyError(
                "Размерность embedding-модели не совпадает с FAISS-индексом. Выполните переиндексацию."
            )
        return cls(
            vector_index,
            chunks,
            BM25Okapi(bm25_data["corpus"], k1=bm25_data["k1"], b=bm25_data["b"]),
            model,
        )

    def search(self, query: str, top_k: int = 6) -> SearchResult:
        vector = self.model.encode([query])
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
