from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    frontend_dir: Path = root / "frontend"
    source_dir: Path = root / "data" / "raw"
    index_dir: Path = root / "data" / "index"
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    reindex_token: str | None = os.getenv("REINDEX_TOKEN") or None


settings = Settings()
