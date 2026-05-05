"""Provider selection from environment variables.

Convention:
- ``STRATEGOS_LLM_PROVIDER`` — explicit selector. One of ``anthropic``,
  ``openai``, ``ollama``. If unset, we auto-detect in this order:
    1. ``ANTHROPIC_API_KEY`` set        -> Anthropic
    2. ``OPENAI_API_KEY`` set           -> OpenAI
    3. ``OLLAMA_HOST`` set OR fallback  -> Ollama
- ``STRATEGOS_LLM_MODEL`` — optional model override.
"""

from __future__ import annotations

import os

from strategos.llm.anthropic_provider import AnthropicProvider
from strategos.llm.base import BaseLLMProvider
from strategos.llm.ollama_provider import OllamaProvider
from strategos.llm.openai_provider import OpenAIProvider


def _resolve_choice() -> str:
    explicit = (os.environ.get("STRATEGOS_LLM_PROVIDER") or "").strip().lower()
    if explicit in ("anthropic", "openai", "ollama"):
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def from_env() -> BaseLLMProvider:
    """Construct a provider from environment variables.

    Raises
    ------
    RuntimeError
        If the requested provider can't be initialised.
    """
    choice = _resolve_choice()
    model_override = os.environ.get("STRATEGOS_LLM_MODEL") or None

    try:
        if choice == "anthropic":
            return AnthropicProvider(model=model_override)
        if choice == "openai":
            return OpenAIProvider(model=model_override)
        return OllamaProvider(model=model_override)
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            f"Could not initialise LLM provider '{choice}': {exc}. "
            "Set STRATEGOS_LLM_PROVIDER and the appropriate credentials "
            "(ANTHROPIC_API_KEY / OPENAI_API_KEY / OLLAMA_HOST)."
        ) from exc
