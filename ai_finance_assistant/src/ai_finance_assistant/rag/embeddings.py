from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from ai_finance_assistant.core.settings import Settings


@lru_cache(maxsize=4)
def _hf_cached(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})


def get_embeddings(settings: Settings) -> Embeddings:
    conf = settings.embeddings
    if conf.provider != "huggingface":
        raise ValueError("Only huggingface embeddings are configured in this MVP.")
    return _hf_cached(conf.model_name)
