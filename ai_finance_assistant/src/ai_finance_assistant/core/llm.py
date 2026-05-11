from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ai_finance_assistant.core.exceptions import ConfigurationError
from ai_finance_assistant.core.settings import EnvSettings, LLMConf


def chat_model_llm(llm_conf: LLMConf, env: EnvSettings) -> BaseChatModel:
    if llm_conf.provider == "openai":
        if not env.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=llm_conf.model,
            temperature=llm_conf.temperature,
            timeout=llm_conf.timeout_seconds,
            api_key=env.openai_api_key,
        )

    if llm_conf.provider == "google_genai":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ConfigurationError("Install langchain-google-genai for Gemini.") from e
        if not env.google_api_key:
            raise ConfigurationError("GOOGLE_API_KEY is not set.")
        return ChatGoogleGenerativeAI(
            model=llm_conf.model,
            temperature=llm_conf.temperature,
            timeout=llm_conf.timeout_seconds,
            google_api_key=env.google_api_key,
        )

    if llm_conf.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ConfigurationError("Install langchain-anthropic for Claude.") from e
        if not env.anthropic_api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(
            model=llm_conf.model,
            temperature=llm_conf.temperature,
            timeout=llm_conf.timeout_seconds,
            anthropic_api_key=env.anthropic_api_key,
        )

    raise ConfigurationError(f"Unknown LLM provider: {llm_conf.provider}")
