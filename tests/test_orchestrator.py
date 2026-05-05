"""End-to-end orchestrator tests."""
from __future__ import annotations

import pytest

from strategos import (
    AuditLog,
    DrawdownStop,
    EqualWeight,
    ExposureCap,
    KillSwitch,
    MajorityVote,
    Orchestrator,
    PortfolioState,
    PositionLimit,
    RuleAgent,
)


def _pf(regime=None, equity_history=None):
    return PortfolioState(
        cash=100_000.0,
        positions={"AAPL": 0.0},
        last_prices={"AAPL": 100.0},
        regime=regime,
        equity_history=list(equity_history or []),
    )


def _trend(name="trend"):
    return RuleAgent(name=name, fn=lambda obs, pf: ("AAPL", 0.6, 0.8))


def _bear(name="bear"):
    return RuleAgent(name=name, fn=lambda obs, pf: ("AAPL", -0.4, 0.5))


def test_orchestrator_end_to_end_two_agents():
    orc = Orchestrator(
        agents=[_trend(), _bear()],
        ensemble=MajorityVote(),
        allocator=EqualWeight(),
        risk_gates=[PositionLimit(0.5), ExposureCap(1.0)],
        audit_log=AuditLog(secret_key=b"\x02" * 32),
    )
    weights = orc.step({}, _pf())
    assert "AAPL" in weights
    assert weights["AAPL"] > 0
    assert abs(weights["AAPL"]) <= 0.5
    assert orc.audit_log.verify_chain() is True


def test_orchestrator_killswitch_zeroes_weights():
    orc = Orchestrator(
        agents=[_trend()],
        ensemble=MajorityVote(),
        allocator=EqualWeight(),
        risk_gates=[KillSwitch(armed=True)],
        audit_log=AuditLog(secret_key=b"\x03" * 32),
    )
    weights = orc.step({}, _pf())
    assert weights == {"AAPL": 0.0}
    assert orc.audit_log.verify_chain() is True


def test_orchestrator_drawdown_blocks_trades():
    orc = Orchestrator(
        agents=[_trend()],
        ensemble=MajorityVote(),
        allocator=EqualWeight(),
        risk_gates=[DrawdownStop(0.1)],
        audit_log=AuditLog(secret_key=b"\x04" * 32),
    )
    pf = _pf(equity_history=[100.0, 120.0, 90.0])
    weights = orc.step({}, pf)
    assert weights["AAPL"] == 0.0
    assert orc.audit_log.verify_chain() is True


def test_orchestrator_swallows_bad_agent():
    class Boom(RuleAgent):
        def act(self, obs, pf):
            raise RuntimeError("agent failed")

    bad = Boom(name="boom", fn=lambda obs, pf: ("AAPL", 0.5, 0.5))
    orc = Orchestrator(
        agents=[bad, _trend()],
        ensemble=MajorityVote(),
        allocator=EqualWeight(),
        risk_gates=[PositionLimit(1.0)],
        audit_log=AuditLog(secret_key=b"\x05" * 32),
    )
    weights = orc.step({}, _pf())
    assert weights["AAPL"] > 0
    assert orc.audit_log.verify_chain() is True
    kinds = [e["kind"] for e in orc.audit_log.all()]
    assert "skip" in kinds


def test_orchestrator_no_decisions_returns_empty():
    class Nothing(RuleAgent):
        def act(self, obs, pf):
            raise ValueError("nope")

    bad = Nothing(name="n", fn=lambda obs, pf: ("AAPL", 0.5, 0.5))
    orc = Orchestrator(
        agents=[bad],
        ensemble=MajorityVote(),
        allocator=EqualWeight(),
        risk_gates=[],
        audit_log=AuditLog(secret_key=b"\x06" * 32),
    )
    weights = orc.step({}, _pf())
    assert weights == {}


def test_orchestrator_llm_advisor_failure_does_not_crash():
    class FlakyAdvisor:
        def post_mortem(self, events):
            raise RuntimeError("LLM down")

    orc = Orchestrator(
        agents=[_trend()],
        ensemble=MajorityVote(),
        allocator=EqualWeight(),
        risk_gates=[PositionLimit(1.0)],
        audit_log=AuditLog(secret_key=b"\x07" * 32),
        llm_advisor=FlakyAdvisor(),
    )
    weights = orc.step({}, _pf())
    assert weights["AAPL"] > 0
    assert orc.audit_log.verify_chain() is True
    kinds = [e["kind"] for e in orc.audit_log.all()]
    assert "system" in kinds


def test_orchestrator_llm_advisor_called_with_events():
    captured = {}

    class Advisor:
        def post_mortem(self, events):
            captured["events"] = list(events)

    orc = Orchestrator(
        agents=[_trend()],
        ensemble=MajorityVote(),
        allocator=EqualWeight(),
        risk_gates=[PositionLimit(1.0)],
        audit_log=AuditLog(secret_key=b"\x08" * 32),
        llm_advisor=Advisor(),
    )
    orc.step({}, _pf())
    assert "events" in captured
    assert any(e["kind"] == "execution" for e in captured["events"])


def test_orchestrator_rejects_duplicate_agent_names():
    with pytest.raises(ValueError):
        Orchestrator(
            agents=[_trend("dup"), _trend("dup")],
            ensemble=MajorityVote(),
            allocator=EqualWeight(),
            risk_gates=[],
        )


def test_orchestrator_requires_at_least_one_agent():
    with pytest.raises(ValueError):
        Orchestrator(
            agents=[],
            ensemble=MajorityVote(),
            allocator=EqualWeight(),
            risk_gates=[],
        )
