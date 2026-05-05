"""Tests for :class:`strategos.llm.advisor.LLMAdvisor`.

All tests use the local ``MockProvider`` from ``test_llm_providers``;
they never touch the network and don't require any vendor SDKs.
"""

from __future__ import annotations

import pytest

from strategos.llm.advisor import LLMAdvisor

from tests.test_llm_providers import MockProvider


@pytest.fixture
def advisor() -> tuple[LLMAdvisor, MockProvider]:
    prov = MockProvider(canned="this is the advisor talking")
    return LLMAdvisor(prov), prov


class TestAdvisorReturnTypes:
    def test_strategy_brief(self, advisor: tuple[LLMAdvisor, MockProvider]) -> None:
        a, _ = advisor
        out = a.strategy_brief(
            agents=["momentum", "meanrev"],
            regime="calm",
            recent_pnl=0.0123,
        )
        assert isinstance(out, str)
        assert out == "this is the advisor talking"

    def test_post_mortem(self, advisor: tuple[LLMAdvisor, MockProvider]) -> None:
        a, _ = advisor
        events = [
            {"ts": 1, "agent": "momentum", "decision": "BUY", "size": 0.1},
            {"ts": 2, "gate": "concentration", "verdict": "PASS"},
        ]
        out = a.post_mortem(events)
        assert isinstance(out, str)
        assert out

    def test_post_mortem_empty(self, advisor: tuple[LLMAdvisor, MockProvider]) -> None:
        a, _ = advisor
        out = a.post_mortem([])
        assert out.startswith("[advisor offline:")

    def test_risk_review(self, advisor: tuple[LLMAdvisor, MockProvider]) -> None:
        a, _ = advisor
        out = a.risk_review(
            decisions=[{"agent": "momentum", "size": 0.5}],
            portfolio={"cash": 1.0, "positions": {"BTC": 0.4}},
        )
        assert isinstance(out, str)
        assert out

    def test_regime_explain(self, advisor: tuple[LLMAdvisor, MockProvider]) -> None:
        a, _ = advisor
        out = a.regime_explain(
            regime="trending_up",
            indicators={"adx": 31.4, "vol_z": -0.7},
        )
        assert isinstance(out, str)
        assert out


class TestAdvisorPromptShape:
    def test_system_prompt_present_and_cacheable(
        self, advisor: tuple[LLMAdvisor, MockProvider]
    ) -> None:
        a, prov = advisor
        a.strategy_brief(agents=["a"], regime="calm", recent_pnl=0.0)
        call = prov.calls[-1]
        assert call["system"] is not None
        assert "senior quantitative risk advisor" in call["system"].lower()
        assert call["cache_system"] is True
        assert call["temperature"] == 0.2

    def test_user_message_contains_inputs(
        self, advisor: tuple[LLMAdvisor, MockProvider]
    ) -> None:
        a, prov = advisor
        a.regime_explain(regime="vol_spike", indicators={"vix": 38})
        call = prov.calls[-1]
        assert len(call["rest"]) == 1
        user_msg = call["rest"][0]
        assert user_msg.role == "user"
        assert "vol_spike" in user_msg.content
        assert "vix" in user_msg.content

    def test_methods_share_same_system_prompt(
        self, advisor: tuple[LLMAdvisor, MockProvider]
    ) -> None:
        """Critical for prompt caching: every advisor call must use the
        identical system prompt so the provider cache hits."""
        a, prov = advisor
        a.strategy_brief(agents=["m"], regime="calm", recent_pnl=0.0)
        a.risk_review(decisions=[], portfolio={})
        a.regime_explain(regime="calm", indicators={})
        systems = {c["system"] for c in prov.calls}
        assert len(systems) == 1


class TestAdvisorErrorHandling:
    def test_provider_exception_returns_offline_string(self) -> None:
        prov = MockProvider()
        prov.raise_exc = RuntimeError("boom")
        a = LLMAdvisor(prov)
        out = a.strategy_brief(agents=["a"], regime=None, recent_pnl=0.0)
        assert out.startswith("[advisor offline:")
        assert "RuntimeError" in out
        assert "boom" in out

    def test_empty_response_is_handled(self) -> None:
        prov = MockProvider(canned="   ")
        a = LLMAdvisor(prov)
        out = a.regime_explain(regime="calm", indicators={})
        assert out.startswith("[advisor offline:")


class TestAdvisorJSONHandling:
    def test_non_serialisable_payload_does_not_crash(self) -> None:
        prov = MockProvider(canned="ok")
        a = LLMAdvisor(prov)

        class Weird:
            def __repr__(self) -> str:
                return "<weird>"

        out = a.risk_review(
            decisions=[{"obj": Weird()}],
            portfolio={"obj": Weird()},
        )
        assert isinstance(out, str)
        assert out == "ok"

    def test_huge_audit_trail_is_truncated(self) -> None:
        prov = MockProvider(canned="ok")
        a = LLMAdvisor(prov)
        events = [{"i": i, "blob": "x" * 200} for i in range(500)]
        out = a.post_mortem(events)
        assert out == "ok"
        user_content = prov.calls[-1]["rest"][0].content
        assert len(user_content) < 20_000
