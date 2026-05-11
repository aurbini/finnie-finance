from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ai_finance_assistant.core.settings import Settings

def build_faiss_index(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


def persist_index(store: FAISS, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))


def load_faiss_index(embeddings: Embeddings, index_dir: Path) -> FAISS | None:
    if not (index_dir / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def write_meta(settings: Settings, chunk_count: int, article_count: int) -> None:
    from ai_finance_assistant.utils.paths import kb_metadata_path

    meta_path = kb_metadata_path(settings)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_count": chunk_count,
        "article_count": article_count,
        "chunk_size": settings.rag.chunk_size,
        "embedding_model": settings.embeddings.model_name,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
