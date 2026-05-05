"""Tests for the LLM provider contract.

These tests must pass without ``anthropic``, ``openai`` or ``ollama``
installed: we exercise the BaseLLMProvider contract through a local
``MockProvider`` and through the concrete providers using *injected*
fake clients.
"""

from __future__ import annotations

from typing import Any

import pytest

from strategos.llm.base import BaseLLMProvider, LLMResponse, Message


class MockProvider(BaseLLMProvider):
    name = "mock"
    default_model = "mock-model-1"

    def __init__(self, canned: str = "ok") -> None:
        self.canned = canned
        self.calls: list[dict[str, Any]] = []
        self.raise_exc: Exception | None = None

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
        if self.raise_exc is not None:
            raise self.raise_exc
        merged_system, rest = self._split_system(messages, system)
        self.calls.append(
            {
                "messages": list(messages),
                "rest": rest,
                "model": model or self.default_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": merged_system,
                "cache_system": cache_system,
            }
        )
        return LLMResponse(
            text=self.canned,
            provider=self.name,
            model=model or self.default_model,
            usage={"input_tokens": 10, "output_tokens": 5, "cached_tokens": 0},
            cost_usd=0.0,
            raw={"echo": self.canned},
        )


class TestMessage:
    def test_valid_roles(self) -> None:
        for role in ("system", "user", "assistant"):
            m = Message(role=role, content="hi")
            assert m.role == role

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValueError):
            Message(role="robot", content="hi")

    def test_non_str_content_raises(self) -> None:
        with pytest.raises(TypeError):
            Message(role="user", content=123)  # type: ignore[arg-type]

    def test_message_is_frozen(self) -> None:
        m = Message(role="user", content="hi")
        with pytest.raises(Exception):
            m.content = "nope"  # type: ignore[misc]


class TestBaseProviderContract:
    def test_returns_llmresponse(self) -> None:
        p = MockProvider(canned="hello")
        resp = p.chat([Message(role="user", content="ping")])
        assert isinstance(resp, LLMResponse)
        assert resp.text == "hello"
        assert resp.provider == "mock"
        assert resp.model == "mock-model-1"
        assert set(resp.usage) >= {"input_tokens", "output_tokens", "cached_tokens"}

    def test_split_system_kwarg_takes_precedence(self) -> None:
        p = MockProvider()
        p.chat(
            [
                Message(role="system", content="from-list"),
                Message(role="user", content="hi"),
            ],
            system="from-kwarg",
        )
        call = p.calls[-1]
        assert call["system"] is not None
        assert call["system"].startswith("from-kwarg")
        assert "from-list" in call["system"]
        assert all(m.role != "system" for m in call["rest"])

    def test_no_system_when_none(self) -> None:
        p = MockProvider()
        p.chat([Message(role="user", content="hi")])
        assert p.calls[-1]["system"] is None

    def test_model_override_passed_through(self) -> None:
        p = MockProvider()
        p.chat([Message(role="user", content="hi")], model="custom")
        assert p.calls[-1]["model"] == "custom"

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseLLMProvider()  # type: ignore[abstract]


class _FakeAnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int, cached: int = 0) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cached


class _FakeAnthropicBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str, usage: _FakeAnthropicUsage) -> None:
        self.content = [_FakeAnthropicBlock(text)]
        self.usage = usage


class _FakeAnthropicClient:
    def __init__(self, text: str = "anthropic-ok") -> None:
        self._text = text
        self.last_kwargs: dict[str, Any] | None = None

        outer = self

        class _Messages:
            def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
                outer.last_kwargs = kwargs
                return _FakeAnthropicResponse(
                    outer._text,
                    _FakeAnthropicUsage(input_tokens=42, output_tokens=7, cached=10),
                )

        self.messages = _Messages()


class TestAnthropicProvider:
    def test_basic_call_and_usage(self) -> None:
        from strategos.llm.anthropic_provider import AnthropicProvider

        fake = _FakeAnthropicClient(text="hi from claude")
        prov = AnthropicProvider(client=fake)

        resp = prov.chat(
            [Message(role="user", content="ping")],
            system="system-prompt",
        )

        assert resp.text == "hi from claude"
        assert resp.provider == "anthropic"
        assert resp.usage == {"input_tokens": 42, "output_tokens": 7, "cached_tokens": 10}
        assert resp.cost_usd is not None and resp.cost_usd >= 0

    def test_system_prompt_uses_cache_control(self) -> None:
        from strategos.llm.anthropic_provider import AnthropicProvider

        fake = _FakeAnthropicClient()
        prov = AnthropicProvider(client=fake)
        prov.chat(
            [Message(role="user", content="hi")],
            system="cache-me",
            cache_system=True,
        )
        assert fake.last_kwargs is not None
        sys_arg = fake.last_kwargs.get("system")
        assert isinstance(sys_arg, list)
        assert sys_arg[0]["text"] == "cache-me"
        assert sys_arg[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_disabled(self) -> None:
        from strategos.llm.anthropic_provider import AnthropicProvider

        fake = _FakeAnthropicClient()
        prov = AnthropicProvider(client=fake)
        prov.chat(
            [Message(role="user", content="hi")],
            system="no-cache",
            cache_system=False,
        )
        sys_arg = fake.last_kwargs["system"]  # type: ignore[index]
        assert "cache_control" not in sys_arg[0]

    def test_no_system_omits_field(self) -> None:
        from strategos.llm.anthropic_provider import AnthropicProvider

        fake = _FakeAnthropicClient()
        prov = AnthropicProvider(client=fake)
        prov.chat([Message(role="user", content="hi")])
        assert "system" not in (fake.last_kwargs or {})


class _FakeOAIPromptDetails:
    def __init__(self, cached: int) -> None:
        self.cached_tokens = cached


class _FakeOAIUsage:
    def __init__(self, prompt: int, completion: int, cached: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _FakeOAIPromptDetails(cached)


class _FakeOAIChoice:
    def __init__(self, text: str) -> None:
        class _M:
            def __init__(self, t: str) -> None:
                self.content = t

        self.message = _M(text)


class _FakeOAIResponse:
    def __init__(self, text: str, usage: _FakeOAIUsage) -> None:
        self.choices = [_FakeOAIChoice(text)]
        self.usage = usage


class _FakeOAICompletions:
    def __init__(self, parent: "_FakeOAIClient") -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> _FakeOAIResponse:
        self._parent.last_kwargs = kwargs
        return _FakeOAIResponse(
            self._parent._text,
            _FakeOAIUsage(prompt=100, completion=20, cached=80),
        )


class _FakeOAIChat:
    def __init__(self, parent: "_FakeOAIClient") -> None:
        self.completions = _FakeOAICompletions(parent)


class _FakeOAIClient:
    def __init__(self, text: str = "openai-ok") -> None:
        self._text = text
        self.last_kwargs: dict[str, Any] | None = None
        self.chat = _FakeOAIChat(self)


class TestOpenAIProvider:
    def test_basic_call_and_cached_tokens(self) -> None:
        from strategos.llm.openai_provider import OpenAIProvider

        fake = _FakeOAIClient(text="hi from gpt")
        prov = OpenAIProvider(client=fake)
        resp = prov.chat(
            [Message(role="user", content="ping")],
            system="sys",
        )
        assert resp.text == "hi from gpt"
        assert resp.provider == "openai"
        assert resp.usage == {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 80}
        assert resp.cost_usd is not None and resp.cost_usd > 0

    def test_system_lifted_to_first_message(self) -> None:
        from strategos.llm.openai_provider import OpenAIProvider

        fake = _FakeOAIClient()
        prov = OpenAIProvider(client=fake)
        prov.chat(
            [Message(role="user", content="hi")],
            system="be-careful",
        )
        msgs = fake.last_kwargs["messages"]  # type: ignore[index]
        assert msgs[0] == {"role": "system", "content": "be-careful"}
        assert msgs[1] == {"role": "user", "content": "hi"}


class _FakeOllamaClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.last_kwargs = kwargs
        return {
            "message": {"role": "assistant", "content": "ollama-ok"},
            "prompt_eval_count": 11,
            "eval_count": 3,
        }


class TestOllamaProvider:
    def test_basic_call(self) -> None:
        from strategos.llm.ollama_provider import OllamaProvider

        fake = _FakeOllamaClient()
        prov = OllamaProvider(client=fake)
        resp = prov.chat(
            [Message(role="user", content="ping")],
            system="sys",
        )
        assert resp.text == "ollama-ok"
        assert resp.provider == "ollama"
        assert resp.cost_usd == 0.0
        assert resp.usage == {"input_tokens": 11, "output_tokens": 3, "cached_tokens": 0}

    def test_system_message_lifted(self) -> None:
        from strategos.llm.ollama_provider import OllamaProvider

        fake = _FakeOllamaClient()
        prov = OllamaProvider(client=fake)
        prov.chat([Message(role="user", content="hi")], system="be-careful")
        msgs = fake.last_kwargs["messages"]  # type: ignore[index]
        assert msgs[0] == {"role": "system", "content": "be-careful"}


class TestFactory:
    def test_explicit_ollama_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strategos.llm.factory import from_env
        from strategos.llm.ollama_provider import OllamaProvider

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("STRATEGOS_LLM_PROVIDER", "ollama")
        prov = from_env()
        assert isinstance(prov, OllamaProvider)

    def test_default_when_no_keys_is_ollama(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strategos.llm.factory import from_env
        from strategos.llm.ollama_provider import OllamaProvider

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("STRATEGOS_LLM_PROVIDER", raising=False)
        prov = from_env()
        assert isinstance(prov, OllamaProvider)

    def test_anthropic_missing_sdk_raises_runtimeerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from strategos.llm.factory import from_env

        monkeypatch.setenv("STRATEGOS_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        try:
            import anthropic  # type: ignore  # noqa: F401
        except ImportError:
            with pytest.raises(RuntimeError):
                from_env()
        else:
            pytest.skip("anthropic SDK is installed; skip missing-SDK path")
