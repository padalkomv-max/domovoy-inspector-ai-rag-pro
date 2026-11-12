from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.rag import build_index


if __name__ == "__main__":
    result = build_index(
        settings.source_dir,
        settings.index_dir,
        settings.embedding_model,
        settings.embedding_revision,
        settings.embedding_cache_dir,
        settings.embedding_max_length,
    )
    print(f"Индексация завершена: {result['documents']} документов, {result['chunks']} чанков, модель {result['model']}")
