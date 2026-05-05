"""Shared types and the abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


VALID_ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class Message:
    """A single chat message exchanged with an LLM.

    Roles follow the OpenAI/Anthropic convention: ``system``, ``user``,
    ``assistant``. Providers are responsible for re-shaping these messages
    into their own native payloads (e.g. Anthropic strips ``system``
    messages from the ``messages`` list and lifts them to a top-level
    ``system`` field).
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"Message.role must be one of {VALID_ROLES}, got {self.role!r}"
            )
        if not isinstance(self.content, str):
            raise TypeError(
                f"Message.content must be str, got {type(self.content).__name__}"
            )


@dataclass(frozen=True)
class LLMResponse:
    """Normalised response from any LLM provider.

    Fields
    ------
    text:
        The assistant's reply as a single string.
    provider:
        Provider name, e.g. ``"anthropic"``, ``"openai"``, ``"ollama"``.
    model:
        Concrete model identifier that produced the response.
    usage:
        Token accounting. Always contains the keys ``input_tokens``,
        ``output_tokens``, ``cached_tokens`` (zero if unknown / unsupported).
    cost_usd:
        Best-effort cost estimate in USD, or ``None`` when unknown.
    raw:
        Native provider payload, retained for debugging.
    """

    text: str
    provider: str
    model: str
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    raw: Any = None


class BaseLLMProvider(ABC):
    """Abstract LLM provider contract.

    Subclasses MUST set ``name`` and ``default_model`` as class attributes
    and implement :meth:`chat`.
    """

    name: str = "base"
    default_model: str = ""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        system: str | None = None,
        cache_system: bool = True,
    ) -> LLMResponse:
        """Send a chat completion request."""
        raise NotImplementedError

    @staticmethod
    def _split_system(
        messages: list[Message], system: str | None
    ) -> tuple[str | None, list[Message]]:
        """Pull any ``role='system'`` messages out of ``messages``.

        If ``system`` is provided as a kwarg, it takes precedence and the
        in-list system messages are concatenated *after* it (newline
        separated). Returns ``(merged_system, non_system_messages)``.
        """
        sys_parts: list[str] = []
        if system:
            sys_parts.append(system)
        rest: list[Message] = []
        for m in messages:
            if m.role == "system":
                sys_parts.append(m.content)
            else:
                rest.append(m)
        merged = "\n\n".join(p for p in sys_parts if p) or None
        return merged, rest
