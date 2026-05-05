"""Allocator invariants."""
from __future__ import annotations

import numpy as np
import pytest

from strategos import (
    Decision,
    EqualWeight,
    KellyOnline,
    PortfolioState,
    RegimeAware,
    RiskParity,
)


def _pf(regime=None):
    return PortfolioState(
        cash=100_000.0,
        positions={"AAPL": 0.0, "MSFT": 0.0},
        last_prices={"AAPL": 100.0, "MSFT": 100.0},
        regime=regime,
    )


def _d(name, asset, pos, conf):
    return Decision(agent_name=name, asset=asset, target_position=pos, confidence=conf)


def test_equal_weight_sums_to_unit_gross():
    decisions = [_d("a", "AAPL", 0.5, 0.9), _d("b", "MSFT", -0.3, 0.6)]
    w = EqualWeight().allocate(decisions, _pf())
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-9)
    assert w["AAPL"] > 0
    assert w["MSFT"] < 0


def test_equal_weight_drops_zero_confidence_or_zero_target():
    decisions = [
        _d("a", "AAPL", 0.5, 0.9),
        _d("b", "MSFT", 0.0, 0.6),
        _d("c", "GOOG", 0.4, 0.0),
    ]
    w = EqualWeight().allocate(decisions, _pf())
    assert "MSFT" not in w
    assert "GOOG" not in w
    assert "AAPL" in w


def test_equal_weight_empty_returns_empty():
    assert EqualWeight().allocate([], _pf()) == {}


def test_risk_parity_inverse_vol():
    rng = np.random.default_rng(0)
    aapl_rets = rng.normal(0, 0.01, 100)
    msft_rets = rng.normal(0, 0.04, 100)
    rp = RiskParity({"AAPL": aapl_rets, "MSFT": msft_rets})
    decisions = [_d("a", "AAPL", 0.5, 1.0), _d("b", "MSFT", 0.5, 1.0)]
    w = rp.allocate(decisions, _pf())
    assert abs(w["AAPL"]) > abs(w["MSFT"])
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-9)


def test_kelly_online_drops_negative_edge():
    decisions = [
        _d("good", "AAPL", 0.5, 0.8),
        _d("bad",  "MSFT", 0.5, 0.4),
    ]
    k = KellyOnline(fraction=0.5)
    w = k.allocate(decisions, _pf())
    assert "AAPL" in w
    assert "MSFT" not in w


def test_kelly_online_respects_unit_gross():
    decisions = [_d("g", "AAPL", 0.7, 0.95), _d("h", "MSFT", -0.6, 0.95)]
    w = KellyOnline(fraction=1.0).allocate(decisions, _pf())
    assert sum(abs(v) for v in w.values()) <= 1.0 + 1e-9


def test_regime_aware_uses_override_when_known():
    rw = {"bull": {"AAPL": 0.7, "MSFT": 0.3}}
    ra = RegimeAware(regime_weights=rw)
    decisions = [_d("a", "AAPL", 0.5, 1.0), _d("b", "MSFT", 0.5, 1.0)]
    w = ra.allocate(decisions, _pf(regime="bull"))
    assert w["AAPL"] > w["MSFT"]
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-9)


def test_regime_aware_falls_back_when_unknown():
    rw = {"bull": {"AAPL": 0.7, "MSFT": 0.3}}
    ra = RegimeAware(regime_weights=rw, fallback=EqualWeight())
    decisions = [_d("a", "AAPL", 0.5, 1.0), _d("b", "MSFT", 0.5, 1.0)]
    w_unknown = ra.allocate(decisions, _pf(regime="chaos"))
    assert abs(abs(w_unknown["AAPL"]) - abs(w_unknown["MSFT"])) < 1e-9
