"""Local Ollama provider.

Talks to a local Ollama daemon (default ``http://localhost:11434``). Tries
the official ``ollama`` Python client first; falls back to a plain
``requests`` POST against ``/api/chat`` so the provider works with only
``requests`` installed.
"""

from __future__ import annotations

import os
from typing import Any

from strategos.llm.base import BaseLLMProvider, LLMResponse, Message


class OllamaProvider(BaseLLMProvider):
    """Provider backed by a local Ollama daemon. Cost is always 0.0."""

    name = "ollama"
    default_model = "llama3.1"

    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        client: Any = None,
        request_timeout: float = 120.0,
    ) -> None:
        self._host = (
            host
            or os.environ.get("OLLAMA_HOST")
            or "http://localhost:11434"
        ).rstrip("/")
        self.default_model = (
            model
            or os.environ.get("OLLAMA_MODEL")
            or self.default_model
        )
        self._timeout = request_timeout

        self._py_client: Any = client
        if self._py_client is None:
            try:
                import ollama  # type: ignore

                self._py_client = ollama.Client(host=self._host)
            except ImportError:
                self._py_client = None

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        system: str | None = None,
        cache_system: bool = True,  # noqa: ARG002
    ) -> LLMResponse:
        merged_system, rest = self._split_system(messages, system)
        chosen_model = model or self.default_model

        api_messages: list[dict[str, str]] = []
        if merged_system:
            api_messages.append({"role": "system", "content": merged_system})
        api_messages.extend({"role": m.role, "content": m.content} for m in rest)

        options = {"temperature": temperature, "num_predict": max_tokens}

        if self._py_client is not None:
            data = self._py_client.chat(
                model=chosen_model,
                messages=api_messages,
                options=options,
                stream=False,
            )
            data_dict = data if isinstance(data, dict) else _to_dict(data)
        else:
            data_dict = self._raw_http_chat(chosen_model, api_messages, options)

        msg = data_dict.get("message") or {}
        text = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        usage = {
            "input_tokens": int(data_dict.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(data_dict.get("eval_count", 0) or 0),
            "cached_tokens": 0,
        }

        return LLMResponse(
            text=text or "",
            provider=self.name,
            model=chosen_model,
            usage=usage,
            cost_usd=0.0,
            raw=data_dict,
        )

    def _raw_http_chat(
        self,
        model: str,
        api_messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OllamaProvider needs either the 'ollama' package or 'requests'. "
                "Install one with: pip install ollama  (or)  pip install requests"
            ) from exc

        payload = {
            "model": model,
            "messages": api_messages,
            "stream": False,
            "options": options,
        }
        r = requests.post(
            f"{self._host}/api/chat", json=payload, timeout=self._timeout
        )
        r.raise_for_status()
        return r.json()


def _to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of an SDK response object to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"message": {"content": str(obj)}}
