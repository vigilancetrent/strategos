"""Ensemble combination correctness."""
from __future__ import annotations

import pytest

from strategos import Decision, PortfolioState
from strategos import MajorityVote, PerformanceWeighted, RegimeConditioned


def _pf(regime=None):
    return PortfolioState(
        cash=100_000.0,
        positions={"AAPL": 0.0},
        last_prices={"AAPL": 100.0},
        regime=regime,
    )


def _d(name, pos, conf, asset="AAPL"):
    return Decision(agent_name=name, asset=asset, target_position=pos, confidence=conf)


def test_majority_vote_takes_higher_confidence():
    decisions = [_d("a", 0.5, 0.9), _d("b", -0.5, 0.1)]
    out = MajorityVote().combine(decisions, _pf())
    assert len(out) == 1
    assert out[0].target_position > 0


def test_majority_vote_zero_when_balanced():
    decisions = [_d("a", 0.5, 0.5), _d("b", -0.5, 0.5)]
    out = MajorityVote().combine(decisions, _pf())
    assert out[0].target_position == pytest.approx(0.0)


def test_performance_weighted_uniform_without_history():
    pw = PerformanceWeighted(window=10)
    decisions = [_d("a", 0.5, 1.0), _d("b", -0.5, 1.0)]
    out = pw.combine(decisions, _pf())
    assert out[0].target_position == pytest.approx(0.0, abs=1e-9)


def test_performance_weighted_reweights_after_loss_streak():
    pw = PerformanceWeighted(window=5, temperature=1.0)
    for _ in range(5):
        pw.record_pnl("a", +1.0)
        pw.record_pnl("b", -1.0)
    decisions = [_d("a", 0.6, 1.0), _d("b", -0.6, 1.0)]
    out = pw.combine(decisions, _pf())
    assert out[0].target_position > 0.1


def test_regime_conditioned_uses_regime_weights():
    rw = {"bull": {"trend": 0.9, "mr": 0.1}, "bear": {"trend": 0.1, "mr": 0.9}}
    rc = RegimeConditioned(regime_to_weights=rw)
    decisions = [_d("trend", 0.5, 1.0), _d("mr", -0.5, 1.0)]

    out_bull = rc.combine(decisions, _pf(regime="bull"))
    out_bear = rc.combine(decisions, _pf(regime="bear"))

    assert out_bull[0].target_position > 0
    assert out_bear[0].target_position < 0


def test_regime_conditioned_unknown_falls_back_uniform():
    rc = RegimeConditioned(regime_to_weights={"bull": {"a": 1.0}})
    decisions = [_d("a", 0.4, 1.0), _d("b", 0.4, 1.0)]
    out = rc.combine(decisions, _pf(regime="???"))
    assert out[0].target_position == pytest.approx(0.4, abs=1e-9)
