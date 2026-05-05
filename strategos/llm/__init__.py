"""LLM advisor sub-package for strategos.

Public API
----------
- ``Message``, ``LLMResponse`` — shared dataclasses (see ``base``).
- ``BaseLLMProvider`` — abstract provider contract.
- ``AnthropicProvider``, ``OpenAIProvider``, ``OllamaProvider`` — concrete providers.
- ``from_env`` — provider factory driven by environment variables.
- ``LLMAdvisor`` — high-level helpers: strategy brief, post-mortem, risk review,
  regime explanation.

Provider SDKs (``anthropic``, ``openai``, ``ollama``) are *lazy-imported* inside
the concrete provider constructors so importing this package never fails on a
machine that has only some — or none — of them installed.
"""

from strategos.llm.base import BaseLLMProvider, LLMResponse, Message
from strategos.llm.anthropic_provider import AnthropicProvider
from strategos.llm.openai_provider import OpenAIProvider
from strategos.llm.ollama_provider import OllamaProvider
from strategos.llm.factory import from_env
from strategos.llm.advisor import LLMAdvisor

__all__ = [
    "Message",
    "LLMResponse",
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "from_env",
    "LLMAdvisor",
]
