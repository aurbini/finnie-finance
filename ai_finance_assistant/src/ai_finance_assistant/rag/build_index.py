from __future__ import annotations

import argparse

from ai_finance_assistant.core.logging_config import configure_logging, get_logger
from ai_finance_assistant.core.settings import load_settings
from ai_finance_assistant.rag.chunker import chunk_documents
from ai_finance_assistant.rag.embeddings import get_embeddings
from ai_finance_assistant.rag.loader import load_markdown_documents
from ai_finance_assistant.rag.vectorstore import build_faiss_index, persist_index, write_meta
from ai_finance_assistant.utils.paths import articles_directory, faiss_index_directory

_log = get_logger(__name__)


def build_index() -> None:
    configure_logging()
    settings = load_settings()
    arts = articles_directory(settings)
    docs = load_markdown_documents(arts)
    if not docs:
        raise SystemExit(
            f"No articles loaded from {arts}. Run scripts/materialize_kb.py first."
        )
    chunks = chunk_documents(docs, settings.rag)
    emb = get_embeddings(settings)
    store = build_faiss_index(chunks, emb)
    out = faiss_index_directory(settings)
    persist_index(store, out)
    write_meta(settings, len(chunks), len(docs))
    _log.info("Indexed %s chunks from %s articles into %s", len(chunks), len(docs), out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local FAISS index for RAG.")
    parser.parse_args()
    build_index()


if __name__ == "__main__":
    main()
