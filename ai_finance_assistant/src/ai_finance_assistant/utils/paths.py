from __future__ import annotations

from pathlib import Path

from ai_finance_assistant.core.exceptions import project_root
from ai_finance_assistant.core.settings import Settings


def articles_directory(settings: Settings) -> Path:
    return project_root() / settings.paths.articles_dir


def faiss_index_directory(settings: Settings) -> Path:
    return project_root() / settings.paths.faiss_index_dir


def kb_metadata_path(settings: Settings) -> Path:
    return project_root() / settings.paths.kb_meta_file
