"""Regime-aware orchestrator: ensemble + allocator both consult portfolio.regime.

Run:
    python examples/02_regime_aware_orchestrator.py
"""
from __future__ import annotations

import numpy as np

from strategos import (
    AuditLog,
    DrawdownStop,
    Orchestrator,
    PortfolioState,
    PositionLimit,
    RegimeAware,
    RegimeConditioned,
    RegimeGate,
    RuleAgent,
)


def trend_fn(obs, portfolio):
    return ("AAPL", 0.7 if obs["ret_5"] > 0 else -0.4, 0.7)


def mr_fn(obs, portfolio):
    if obs["ret_1"] > 0.02:
        return ("AAPL", -0.6, 0.7)
    if obs["ret_1"] < -0.02:
        return ("AAPL", 0.6, 0.7)
    return ("AAPL", 0.0, 0.1)


def main() -> None:
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.015, 80)))

    ensemble = RegimeConditioned(
        regime_to_weights={
            "bull": {"trend": 0.8, "mr": 0.2},
            "bear": {"trend": 0.2, "mr": 0.8},
            "chop": {"trend": 0.4, "mr": 0.6},
        },
    )
    allocator = RegimeAware(
        regime_weights={
            "bull": {"AAPL": 0.8},
            "bear": {"AAPL": 0.4},
            "chop": {"AAPL": 0.5},
        },
    )

    orc = Orchestrator(
        agents=[RuleAgent("trend", trend_fn), RuleAgent("mr", mr_fn)],
        ensemble=ensemble,
        allocator=allocator,
        risk_gates=[
            PositionLimit(0.7),
            DrawdownStop(0.25),
            RegimeGate(allowed={"bull", "bear", "chop"}, allow_unknown=False),
        ],
        audit_log=AuditLog(secret_key=b"regime-aware-demo" * 2),
    )

    pf = PortfolioState(
        cash=100_000.0,
        positions={"AAPL": 0.0},
        last_prices={"AAPL": float(prices[5])},
        equity_history=[100_000.0],
        regime="bull",
    )

    for t in range(5, len(prices)):
        ret_1 = (prices[t] - prices[t - 1]) / prices[t - 1]
        ret_5 = (prices[t] - prices[t - 5]) / prices[t - 5]
        if ret_5 > 0.02:
            pf.regime = "bull"
        elif ret_5 < -0.02:
            pf.regime = "bear"
        else:
            pf.regime = "chop"
        pf.last_prices["AAPL"] = float(prices[t])
        weights = orc.step({"ret_1": float(ret_1), "ret_5": float(ret_5)}, pf)

        if t + 1 < len(prices):
            next_ret = (prices[t + 1] - prices[t]) / prices[t]
            pnl = weights.get("AAPL", 0.0) * next_ret * pf.equity
            pf.cash += pnl
            pf.equity_history.append(pf.equity)

    print(f"Final equity:          {pf.equity:,.2f}")
    print(f"Final regime:          {pf.regime}")
    print(f"Audit events recorded: {len(orc.audit_log)}")
    print(f"Audit chain intact:    {orc.audit_log.verify_chain()}")


if __name__ == "__main__":
    main()
