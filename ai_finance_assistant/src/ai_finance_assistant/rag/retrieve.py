from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ai_finance_assistant.core.settings import Settings


@dataclass
class RetrievedChunk:
    content: str
    title: str
    category: str
    source_id: str


def format_context_for_prompt(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[Source {i}] {c.title} ({c.category}, id={c.source_id})\n{c.content.strip()}"
        )
    return "\n\n".join(lines)


def format_sources_markdown(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(f"- **{c.title}** ({c.category}) — `{c.source_id}`" for c in chunks)


def retrieve_similar(
    store: FAISS,
    settings: Settings,
    query: str,
    *,
    categories: list[str] | None = None,
    k: int | None = None,
) -> tuple[list[RetrievedChunk], list[Document]]:
    k = k or settings.rag.top_k_default

    docs_scores = store.similarity_search_with_score(query, k=k * 6 if categories else k)
    collected: list[Document] = []
    for doc, _score in docs_scores:
        if categories:
            cat = doc.metadata.get("category", "")
            if cat not in categories:
                continue
        collected.append(doc)
        if len(collected) >= k:
            break

    chunks = [
        RetrievedChunk(
            content=d.page_content,
            title=str(d.metadata.get("title", "unknown")),
            category=str(d.metadata.get("category", "unknown")),
            source_id=str(d.metadata.get("source_id", "?")),
        )
        for d in collected[:k]
    ]
    return chunks, collected[:k]
