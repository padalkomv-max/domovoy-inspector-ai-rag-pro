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
        "EMBEDDING_MODEL", "sentence-transformers/static-similarity-mrl-multilingual-v1"
    )
    embedding_revision: str = os.getenv("EMBEDDING_MODEL_REVISION", "b68f4122911bcffcd6e1f695f2d99cd6788972d8")
    embedding_cache_dir: str | None = os.getenv("EMBEDDING_CACHE_DIR") or None
    embedding_max_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    reindex_token: str | None = os.getenv("REINDEX_TOKEN") or None


settings = Settings()
