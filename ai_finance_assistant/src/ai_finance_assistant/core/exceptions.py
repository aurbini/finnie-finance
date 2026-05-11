from __future__ import annotations

from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when config or environment prevents the app from starting."""


def package_root() -> Path:
    """The installed `ai_finance_assistant` package directory."""
    # .../src/ai_finance_assistant/core/exceptions.py -> package is two levels up from `core`.
    return Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Directory that contains ``config.yaml`` (the checkout's ``ai_finance_assistant`` folder)."""
    return Path(__file__).resolve().parents[3]
