"""Shared dataclasses — the single source of truth for strategos types."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class Decision:
    """An agent's recommendation for a single asset at a single timestep."""

    agent_name: str
    asset: str
    target_position: float
    confidence: float
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (-1.0 - 1e-9 <= self.target_position <= 1.0 + 1e-9):
            raise ValueError(
                f"Decision.target_position must lie in [-1, 1], got {self.target_position}"
            )
        if not (0.0 - 1e-9 <= self.confidence <= 1.0 + 1e-9):
            raise ValueError(
                f"Decision.confidence must lie in [0, 1], got {self.confidence}"
            )


@dataclass(frozen=True)
class Verdict:
    """A risk gate's review of a candidate decision."""

    ok: bool
    reason: str
    adjusted_position: Optional[float] = None


@dataclass
class PortfolioState:
    """Mutable snapshot of portfolio state passed to agents and gates."""

    cash: float
    positions: dict[str, float]
    last_prices: dict[str, float]
    equity_history: list[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    regime: str | None = None

    @property
    def equity(self) -> float:
        mtm = sum(
            self.positions.get(a, 0.0) * self.last_prices.get(a, 0.0)
            for a in self.positions
        )
        return self.cash + mtm

    @property
    def gross_exposure(self) -> float:
        eq = self.equity
        if eq <= 0:
            return 0.0
        gross = sum(
            abs(self.positions.get(a, 0.0) * self.last_prices.get(a, 0.0))
            for a in self.positions
        )
        return gross / eq

    def drawdown(self) -> float:
        if not self.equity_history:
            return 0.0
        peak = max(self.equity_history)
        if peak <= 0:
            return 0.0
        last = self.equity_history[-1]
        return max(0.0, (peak - last) / peak)


@dataclass(frozen=True)
class AuditEvent:
    """A single hash-chained audit-log entry."""

    seq: int
    timestamp: str
    kind: str
    payload: dict
    prev_hash: str
    hash: str


__all__ = ["Decision", "Verdict", "PortfolioState", "AuditEvent"]
