"""Two-agent ensemble: a trend follower and a mean-reverter on a single asset.

Run from the repo root:
    python examples/01_two_agent_ensemble.py
"""
from __future__ import annotations

import numpy as np

from strategos import (
    AuditLog,
    EqualWeight,
    ExposureCap,
    MajorityVote,
    Orchestrator,
    PortfolioState,
    PositionLimit,
    RuleAgent,
)


def trend_fn(obs, portfolio):
    if obs["ret_5"] > 0:
        return ("AAPL", 0.6, 0.7)
    return ("AAPL", -0.3, 0.5)


def mean_revert_fn(obs, portfolio):
    if obs["ret_1"] > 0.02:
        return ("AAPL", -0.5, 0.6)
    if obs["ret_1"] < -0.02:
        return ("AAPL", 0.5, 0.6)
    return ("AAPL", 0.0, 0.1)


def main() -> None:
    rng = np.random.default_rng(42)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, 60)))

    orc = Orchestrator(
        agents=[
            RuleAgent("trend", trend_fn),
            RuleAgent("mean_revert", mean_revert_fn),
        ],
        ensemble=MajorityVote(),
        allocator=EqualWeight(),
        risk_gates=[PositionLimit(0.5), ExposureCap(1.0)],
        audit_log=AuditLog(secret_key=b"example-key" * 4),
    )

    pf = PortfolioState(
        cash=100_000.0,
        positions={"AAPL": 0.0},
        last_prices={"AAPL": float(prices[5])},
        equity_history=[100_000.0],
    )

    for t in range(5, len(prices)):
        ret_1 = (prices[t] - prices[t - 1]) / prices[t - 1]
        ret_5 = (prices[t] - prices[t - 5]) / prices[t - 5]
        pf.last_prices["AAPL"] = float(prices[t])
        weights = orc.step({"ret_1": float(ret_1), "ret_5": float(ret_5)}, pf)
        if t + 1 < len(prices):
            next_ret = (prices[t + 1] - prices[t]) / prices[t]
            pnl = weights.get("AAPL", 0.0) * next_ret * pf.equity
            pf.cash += pnl
            pf.equity_history.append(pf.equity)

    print(f"Final equity:          {pf.equity:,.2f}")
    print(f"Audit events recorded: {len(orc.audit_log)}")
    print(f"Audit chain intact:    {orc.audit_log.verify_chain()}")


if __name__ == "__main__":
    main()
