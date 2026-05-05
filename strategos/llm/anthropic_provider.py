"""Anthropic Claude provider with prompt caching support."""

from __future__ import annotations

import os
from typing import Any

from strategos.llm.base import BaseLLMProvider, LLMResponse, Message


# Indicative pricing in USD per 1M tokens. Conservative estimates.
_PRICING_PER_MTOK = {
    # model_id: (input, output, cached_input)
    "claude-opus-4-7": (15.0, 75.0, 1.50),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-haiku-4-5-20251001": (0.80, 4.0, 0.08),
}


class AnthropicProvider(BaseLLMProvider):
    """Provider backed by ``anthropic.Anthropic`` (Messages API).

    Uses ephemeral prompt caching for the system prompt when
    ``cache_system=True`` is passed to :meth:`chat`, which dramatically
    reduces cost & latency when the advisor's static system prompt is
    reused across many calls in a session.
    """

    name = "anthropic"
    default_model = "claude-opus-4-7"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "AnthropicProvider requires the 'anthropic' package. "
                    "Install it with: pip install anthropic"
                ) from exc

            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY env var not set and no api_key passed"
                )
            self._client = anthropic.Anthropic(api_key=key)
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
        cache_system: bool = True,
    ) -> LLMResponse:
        merged_system, rest = self._split_system(messages, system)
        chosen_model = model or self.default_model

        system_arg: Any
        if merged_system:
            block: dict[str, Any] = {"type": "text", "text": merged_system}
            if cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            system_arg = [block]
        else:
            system_arg = None

        api_messages = [{"role": m.role, "content": m.content} for m in rest]

        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_arg is not None:
            kwargs["system"] = system_arg

        resp = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        for block in getattr(resp, "content", []) or []:
            t = getattr(block, "text", None)
            if t:
                text_parts.append(t)
        text = "".join(text_parts)

        usage_obj = getattr(resp, "usage", None)
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0,
            "cached_tokens": getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
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
