"""OpenAI provider (Chat Completions API).

OpenAI performs prompt caching automatically for any prompt >= 1024 tokens,
so ``cache_system`` is honoured implicitly — there is no special syntax for
us to emit. We surface the cached-token count back to the caller via
``LLMResponse.usage["cached_tokens"]`` when the API reports it.
"""

from __future__ import annotations

import os
from typing import Any

from strategos.llm.base import BaseLLMProvider, LLMResponse, Message


_PRICING_PER_MTOK = {
    "gpt-4.1": (5.0, 15.0, 1.25),
    "gpt-4o": (2.50, 10.0, 1.25),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
}


class OpenAIProvider(BaseLLMProvider):
    """Provider backed by ``openai.OpenAI``."""

    name = "openai"
    default_model = "gpt-4.1"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import openai  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "OpenAIProvider requires the 'openai' package. "
                    "Install it with: pip install openai"
                ) from exc

            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY env var not set and no api_key passed"
                )
            self._client = openai.OpenAI(api_key=key)
        else:
            self._client = client

        self.default_model = model or self.default_model

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        system: str | None = None,
        cache_system: bool = True,  # noqa: ARG002 - implicit on OpenAI
    ) -> LLMResponse:
        merged_system, rest = self._split_system(messages, system)
        chosen_model = model or self.default_model

        api_messages: list[dict[str, str]] = []
        if merged_system:
            api_messages.append({"role": "system", "content": merged_system})
        api_messages.extend({"role": m.role, "content": m.content} for m in rest)

        resp = self._client.chat.completions.create(
            model=chosen_model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choices = getattr(resp, "choices", []) or []
        text = ""
        if choices:
            msg = getattr(choices[0], "message", None)
            text = getattr(msg, "content", None) or ""

        usage_obj = getattr(resp, "usage", None)
        input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0

        cached_tokens = 0
        details = getattr(usage_obj, "prompt_tokens_details", None)
        if details is not None:
            cached_tokens = (
                getattr(details, "cached_tokens", None)
                if hasattr(details, "cached_tokens")
                else (details.get("cached_tokens", 0) if isinstance(details, dict) else 0)
            ) or 0

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
        }

        cost = self._estimate_cost(chosen_model, usage)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=chosen_model,
            usage=usage,
            cost_usd=cost,
            raw=resp,
        )

    @staticmethod
    def _estimate_cost(model: str, usage: dict) -> float | None:
        prices = _PRICING_PER_MTOK.get(model)
        if not prices:
            return None
        in_price, out_price, cache_price = prices
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        cached = usage.get("cached_tokens", 0)
        billed_input = max(in_tok - cached, 0)
        return (
            (billed_input / 1_000_000.0) * in_price
            + (cached / 1_000_000.0) * cache_price
            + (out_tok / 1_000_000.0) * out_price
        )
