"""Risk-gate invariants and composition behaviour."""
from __future__ import annotations

import pytest

from strategos import (
    Decision,
    DrawdownStop,
    ExposureCap,
    KillSwitch,
    PortfolioState,
    PositionLimit,
    RegimeGate,
    VaRLimit,
    compose,
)


def _decision(pos=0.8, asset="AAPL"):
    return Decision(agent_name="x", asset=asset, target_position=pos, confidence=1.0)


def _pf(equity_history=None, regime=None, positions=None, prices=None, cash=100_000.0):
    return PortfolioState(
        cash=cash,
        positions=positions or {"AAPL": 0.0},
        last_prices=prices or {"AAPL": 100.0},
        equity_history=list(equity_history or []),
        regime=regime,
    )


def test_position_limit_shrinks_not_enlarges():
    gate = PositionLimit(0.3)
    v = gate.review(_decision(0.8), _pf())
    assert v.ok
    assert v.adjusted_position == pytest.approx(0.3)
    v_neg = gate.review(_decision(-0.9), _pf())
    assert v_neg.adjusted_position == pytest.approx(-0.3)


def test_position_limit_passthrough_when_within():
    v = PositionLimit(0.5).review(_decision(0.2), _pf())
    assert v.adjusted_position == pytest.approx(0.2)


def test_drawdown_stop_blocks_above_threshold():
    pf = _pf(equity_history=[100.0, 110.0, 80.0])
    gate = DrawdownStop(0.25)
    v = gate.review(_decision(0.5), pf)
    assert not v.ok
    assert v.adjusted_position == 0.0


def test_drawdown_stop_passes_below_threshold():
    pf = _pf(equity_history=[100.0, 110.0, 105.0])
    gate = DrawdownStop(0.25)
    v = gate.review(_decision(0.5), pf)
    assert v.ok
    assert v.adjusted_position == pytest.approx(0.5)


def test_killswitch_armed_blocks_everything():
    ks = KillSwitch(armed=True)
    v = ks.review(_decision(0.4), _pf())
    assert not v.ok
    assert v.adjusted_position == 0.0
    ks.disarm()
    assert ks.review(_decision(0.4), _pf()).ok


def test_regime_gate_allowed_set():
    g = RegimeGate({"bull"})
    assert g.review(_decision(0.3), _pf(regime="bull")).ok
    assert not g.review(_decision(0.3), _pf(regime="bear")).ok
    assert not g.review(_decision(0.3), _pf(regime=None)).ok
    g2 = RegimeGate({"bull"}, allow_unknown=True)
    assert g2.review(_decision(0.3), _pf(regime=None)).ok


def test_var_limit_insufficient_history_passes():
    pf = _pf(equity_history=[100.0, 101.0])
    g = VaRLimit(var_threshold=0.05)
    v = g.review(_decision(0.5), pf)
    assert v.ok
    assert v.adjusted_position == pytest.approx(0.5)


def test_var_limit_shrinks_when_breached():
    eq = [100.0]
    rng_pattern = [-0.10, 0.01, -0.12, 0.02, -0.15, 0.03, -0.08, 0.0, -0.20, 0.01] * 4
    for r in rng_pattern:
        eq.append(eq[-1] * (1 + r))
    g = VaRLimit(var_threshold=0.02, alpha=0.05, min_history=10)
    v = g.review(_decision(0.8), _pf(equity_history=eq))
    assert v.ok
    assert abs(v.adjusted_position) < 0.8


def test_exposure_cap_respects_existing_positions():
    positions = {"MSFT": 500.0, "AAPL": 0.0}
    prices = {"MSFT": 100.0, "AAPL": 100.0}
    pf = _pf(positions=positions, prices=prices, cash=50_000.0)
    g = ExposureCap(0.6)
    v = g.review(_decision(0.5, asset="AAPL"), pf)
    assert v.adjusted_position == pytest.approx(0.1, abs=1e-6)


def test_compose_shrinks_monotonically():
    chain = compose([PositionLimit(0.5), PositionLimit(0.3), PositionLimit(0.1)])
    v = chain.review(_decision(0.9), _pf())
    assert v.ok
    assert v.adjusted_position == pytest.approx(0.1)


def test_compose_propagates_veto():
    chain = compose([PositionLimit(0.5), KillSwitch(armed=True), PositionLimit(0.1)])
    v = chain.review(_decision(0.9), _pf())
    assert not v.ok
    assert v.adjusted_position == 0.0


def test_compose_preserves_sign():
    chain = compose([PositionLimit(0.4), PositionLimit(0.2)])
    v = chain.review(_decision(-0.9), _pf())
    assert v.ok
    assert v.adjusted_position == pytest.approx(-0.2)
