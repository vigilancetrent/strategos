"""Composable risk gates."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from strategos.types import Decision, PortfolioState, Verdict


class RiskGate(ABC):
    """Base class for risk gates."""

    name: str = "risk_gate"

    @abstractmethod
    def review(self, decision: Decision, portfolio: PortfolioState) -> Verdict:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} name={self.name!r}>"


class PositionLimit(RiskGate):
    """Cap ``|target_position|`` at ``max_position``."""

    def __init__(self, max_position: float) -> None:
        if max_position < 0:
            raise ValueError("max_position must be >= 0")
        self.max_position = float(max_position)
        self.name = "position_limit"

    def review(self, decision: Decision, portfolio: PortfolioState) -> Verdict:
        target = decision.target_position
        if abs(target) <= self.max_position:
            return Verdict(ok=True, reason="within_position_limit", adjusted_position=target)
        adjusted = float(np.sign(target) * self.max_position)
        return Verdict(
            ok=True,
            reason=f"shrunk_to_position_limit({self.max_position})",
            adjusted_position=adjusted,
        )


class ExposureCap(RiskGate):
    """Cap total gross exposure across the portfolio at ``max_gross``."""

    def __init__(self, max_gross: float) -> None:
        if max_gross < 0:
            raise ValueError("max_gross must be >= 0")
        self.max_gross = float(max_gross)
        self.name = "exposure_cap"

    def review(self, decision: Decision, portfolio: PortfolioState) -> Verdict:
        other_gross = 0.0
        eq = portfolio.equity
        if eq <= 0:
            return Verdict(ok=False, reason="non_positive_equity", adjusted_position=0.0)
        for asset, shares in portfolio.positions.items():
            if asset == decision.asset:
                continue
            px = portfolio.last_prices.get(asset, 0.0)
            other_gross += abs(shares * px) / eq

        remaining = max(0.0, self.max_gross - other_gross)
        target = decision.target_position
        if abs(target) <= remaining:
            return Verdict(ok=True, reason="within_exposure_cap", adjusted_position=target)
        adjusted = float(np.sign(target) * remaining)
        return Verdict(
            ok=True,
            reason=f"shrunk_to_exposure_cap({self.max_gross})",
            adjusted_position=adjusted,
        )


class DrawdownStop(RiskGate):
    """Veto everything when current drawdown exceeds ``max_dd``."""

    def __init__(self, max_dd: float) -> None:
        if not (0 < max_dd <= 1):
            raise ValueError("max_dd must lie in (0, 1]")
        self.max_dd = float(max_dd)
        self.name = "drawdown_stop"

    def review(self, decision: Decision, portfolio: PortfolioState) -> Verdict:
        dd = portfolio.drawdown()
        if dd >= self.max_dd:
            return Verdict(
                ok=False,
                reason=f"drawdown_breach({dd:.4f}>={self.max_dd:.4f})",
                adjusted_position=0.0,
            )
        return Verdict(
            ok=True,
            reason=f"drawdown_ok({dd:.4f}<{self.max_dd:.4f})",
            adjusted_position=decision.target_position,
        )


class VaRLimit(RiskGate):
    """Historic Value-at-Risk gate over ``portfolio.equity_history``."""

    def __init__(self, var_threshold: float, alpha: float = 0.05, min_history: int = 20) -> None:
        if var_threshold <= 0:
            raise ValueError("var_threshold must be > 0")
        if not (0 < alpha < 0.5):
            raise ValueError("alpha must lie in (0, 0.5)")
        if min_history < 2:
            raise ValueError("min_history must be >= 2")
        self.var_threshold = float(var_threshold)
        self.alpha = float(alpha)
        self.min_history = int(min_history)
        self.name = "var_limit"

    def _historic_var(self, equity_history: list[float]) -> float | None:
        if len(equity_history) < self.min_history:
            return None
        eq = np.asarray(equity_history, dtype=float)
        rets = np.diff(eq) / np.where(eq[:-1] == 0.0, 1.0, eq[:-1])
        if rets.size == 0:
            return None
        q = np.quantile(rets, self.alpha)
        return float(abs(min(q, 0.0)))

    def review(self, decision: Decision, portfolio: PortfolioState) -> Verdict:
        var = self._historic_var(portfolio.equity_history)
        if var is None:
            return Verdict(
                ok=True,
                reason="var_insufficient_history",
                adjusted_position=decision.target_position,
            )
        if var <= self.var_threshold:
            return Verdict(
                ok=True,
                reason=f"var_ok({var:.4f}<={self.var_threshold:.4f})",
                adjusted_position=decision.target_position,
            )
        ratio = self.var_threshold / var if var > 0 else 0.0
        adjusted = float(decision.target_position * ratio)
        return Verdict(
            ok=True,
            reason=f"var_shrunk(ratio={ratio:.4f})",
            adjusted_position=adjusted,
        )


class KillSwitch(RiskGate):
    """Manual veto. When ``armed`` is True, every decision is blocked."""

    def __init__(self, armed: bool = False) -> None:
        self.armed = bool(armed)
        self.name = "kill_switch"

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def review(self, decision: Decision, portfolio: PortfolioState) -> Verdict:
        if self.armed:
            return Verdict(ok=False, reason="kill_switch_armed", adjusted_position=0.0)
        return Verdict(ok=True, reason="kill_switch_disarmed", adjusted_position=decision.target_position)


class RegimeGate(RiskGate):
    """Permit decisions only when ``portfolio.regime`` is in ``allowed``."""

    def __init__(self, allowed: set[str] | Iterable[str], allow_unknown: bool = False) -> None:
        self.allowed: set[str] = set(allowed)
        self.allow_unknown = bool(allow_unknown)
        self.name = "regime_gate"

    def review(self, decision: Decision, portfolio: PortfolioState) -> Verdict:
        regime = portfolio.regime
        if regime is None:
            if self.allow_unknown:
                return Verdict(
                    ok=True,
                    reason="regime_unknown_allowed",
                    adjusted_position=decision.target_position,
                )
            return Verdict(ok=False, reason="regime_unknown_blocked", adjusted_position=0.0)
        if regime in self.allowed:
            return Verdict(
                ok=True,
                reason=f"regime_allowed({regime})",
                adjusted_position=decision.target_position,
            )
        return Verdict(ok=False, reason=f"regime_blocked({regime})", adjusted_position=0.0)


class ComposedGate(RiskGate):
    """Chain of gates; vetoes propagate, sizes shrink monotonically."""

    def __init__(self, gates: Iterable[RiskGate]) -> None:
        self._gates: list[RiskGate] = list(gates)
        self.name = "composed[" + ",".join(g.name for g in self._gates) + "]"

    @property
    def gates(self) -> list[RiskGate]:
        return list(self._gates)

    def review(self, decision: Decision, portfolio: PortfolioState) -> Verdict:
        original_sign = float(np.sign(decision.target_position))
        current_pos = decision.target_position
        reasons: list[str] = []
        for gate in self._gates:
            candidate = Decision(
                agent_name=decision.agent_name,
                asset=decision.asset,
                target_position=current_pos,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                metadata=decision.metadata,
            )
            verdict = gate.review(candidate, portfolio)
            reasons.append(f"{gate.name}:{verdict.reason}")
            if not verdict.ok:
                return Verdict(
                    ok=False,
                    reason=" -> ".join(reasons),
                    adjusted_position=0.0,
                )
            new_pos = (
                verdict.adjusted_position
                if verdict.adjusted_position is not None
                else current_pos
            )
            if abs(new_pos) > abs(current_pos) + 1e-12:
                new_pos = float(np.sign(new_pos) * abs(current_pos))
            if original_sign != 0 and np.sign(new_pos) not in (0.0, original_sign):
                new_pos = 0.0
            current_pos = new_pos
        return Verdict(ok=True, reason=" -> ".join(reasons), adjusted_position=current_pos)


def compose(gates: Iterable[RiskGate]) -> ComposedGate:
    """Compose a sequence of gates into a single :class:`ComposedGate`."""
    return ComposedGate(gates)


__all__ = [
    "RiskGate",
    "PositionLimit",
    "ExposureCap",
    "DrawdownStop",
    "VaRLimit",
    "KillSwitch",
    "RegimeGate",
    "ComposedGate",
    "compose",
]
