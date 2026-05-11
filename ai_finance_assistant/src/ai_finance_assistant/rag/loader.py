from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.documents import Document

from ai_finance_assistant.core.logging_config import get_logger

_log = get_logger(__name__)


@dataclass
class ParsedArticle:
    path: Path
    title: str
    category: str
    source_id: str
    body: str


def _split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    raw = raw.lstrip("\ufeff")
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    meta_yaml = parts[1]
    body = parts[2].strip()
    meta = yaml.safe_load(meta_yaml) or {}
    return {str(k): str(v) for k, v in meta.items()}, body


def load_markdown_documents(articles_root: Path) -> list[Document]:
    docs: list[Document] = []
    if not articles_root.exists():
        _log.warning("Articles dir missing at %s", articles_root)
        return docs
    for md in sorted(articles_root.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        meta, body = _split_frontmatter(text)
        title = meta.get("title", md.stem)
        category = meta.get("category", "investing_basics")
        source_id = meta.get("source_id", md.stem)
        try:
            rel_path = md.relative_to(articles_root)
        except ValueError:
            rel_path = md
        docs.append(
            Document(
                page_content=body,
                metadata={
                    "title": title,
                    "category": category,
                    "source_id": source_id,
                    "path": str(rel_path),
                },
            )
        )
    _log.info("Loaded %s markdown docs from %s", len(docs), articles_root)
    return docs
