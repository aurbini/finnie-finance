from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_finance_assistant.core.exceptions import ConfigurationError


class Paths(BaseModel):
    articles_dir: str = "data/articles"
    faiss_index_dir: str = "data/index"
    kb_meta_file: str = "data/index/metadata.json"


class LLMConf(BaseModel):
    provider: Literal["openai", "google_genai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.25
    timeout_seconds: int = 60


class EmbeddingsConf(BaseModel):
    provider: Literal["huggingface"] = "huggingface"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class RAGConf(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k_default: int = 4
    categories: list[str] = Field(default_factory=list)


class MarketConf(BaseModel):
    quote_cache_ttl_seconds: int = 90
    history_days_default: int = 30
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"


class AppConf(BaseModel):
    educational_disclaimer: str = ""


class Settings(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    llm: LLMConf = Field(default_factory=LLMConf)
    embeddings: EmbeddingsConf = Field(default_factory=EmbeddingsConf)
    rag: RAGConf = Field(default_factory=RAGConf)
    market: MarketConf = Field(default_factory=MarketConf)
    app: AppConf = Field(default_factory=AppConf)


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    alpha_vantage_api_key: str | None = None


def _merge_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(config_path: Path | None = None) -> Settings:
    load_dotenv()
    root = config_path or _default_config_path()
    if not root.exists():
        raise ConfigurationError(f"Missing config file: {root}")
    data = yaml.safe_load(root.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(data)


def _default_config_path() -> Path:
    # When running from project root (ai_finance_assistant/)
    here = Path.cwd()
    cand = here / "config.yaml"
    if cand.exists():
        return cand
    # When cwd is repo root (finnie-finance)
    cand2 = here / "ai_finance_assistant" / "config.yaml"
    if cand2.exists():
        return cand2
    # Package-relative project root
    from ai_finance_assistant.core.exceptions import package_root

    pkg = package_root()
    pr = pkg.parents[2]
    return pr / "config.yaml"


def load_env() -> EnvSettings:
    return EnvSettings()
