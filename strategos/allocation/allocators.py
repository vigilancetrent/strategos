"""Capital allocators."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from strategos.types import Decision, PortfolioState


def _normalise_to_unit_gross(weights: dict[str, float]) -> dict[str, float]:
    """Scale weights so that sum(|w|) <= 1, preserving signs and ratios."""
    if not weights:
        return {}
    gross = sum(abs(w) for w in weights.values())
    if gross <= 1.0 + 1e-12:
        return {k: float(v) for k, v in weights.items()}
    scale = 1.0 / gross
    return {k: float(v * scale) for k, v in weights.items()}


class Allocator(ABC):
    name: str = "allocator"

    @abstractmethod
    def allocate(
        self,
        decisions: list[Decision],
        portfolio: PortfolioState,
    ) -> dict[str, float]:
        raise NotImplementedError


class EqualWeight(Allocator):
    """Split capital equally across decisions with non-zero confidence AND non-zero target."""

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = float(eps)
        self.name = "equal_weight"

    def allocate(self, decisions: list[Decision], portfolio: PortfolioState) -> dict[str, float]:
        active = [
            d for d in decisions
            if d.confidence > self.eps and abs(d.target_position) > self.eps
        ]
        if not active:
            return {}
        share = 1.0 / len(active)
        weights: dict[str, float] = {}
        for d in active:
            weights[d.asset] = weights.get(d.asset, 0.0) + float(np.sign(d.target_position) * share)
        return _normalise_to_unit_gross(weights)


class RiskParity(Allocator):
    """Inverse-volatility allocation."""

    def __init__(
        self,
        lookback_returns: dict[str, np.ndarray],
        eps: float = 1e-9,
        floor_vol: float = 1e-6,
    ) -> None:
        self._returns = {k: np.asarray(v, dtype=float) for k, v in lookback_returns.items()}
        self.eps = float(eps)
        self.floor_vol = float(floor_vol)
        self.name = "risk_parity"

    def update_returns(self, asset: str, returns: np.ndarray) -> None:
        self._returns[asset] = np.asarray(returns, dtype=float)

    def allocate(self, decisions: list[Decision], portfolio: PortfolioState) -> dict[str, float]:
        active = [d for d in decisions if abs(d.target_position) > self.eps]
        if not active:
            return {}
        inv_vols: dict[str, float] = {}
        for d in active:
            r = self._returns.get(d.asset)
            if r is None or r.size < 2:
                vol = self.floor_vol
            else:
                vol = float(np.std(r, ddof=1))
                if not np.isfinite(vol) or vol < self.floor_vol:
                    vol = self.floor_vol
            inv_vols[d.asset] = 1.0 / vol
        total = sum(inv_vols.values())
        if total <= 0:
            return {}
        weights: dict[str, float] = {}
        for d in active:
            w = inv_vols[d.asset] / total
            weights[d.asset] = weights.get(d.asset, 0.0) + float(np.sign(d.target_position) * w)
        return _normalise_to_unit_gross(weights)


class KellyOnline(Allocator):
    """Half-Kelly using each agent's confidence as a proxy for win probability."""

    def __init__(self, fraction: float = 0.5, eps: float = 1e-9) -> None:
        if not (0 < fraction <= 1):
            raise ValueError("fraction must lie in (0, 1]")
        self.fraction = float(fraction)
        self.eps = float(eps)
        self.name = "kelly_online"

    def allocate(self, decisions: list[Decision], portfolio: PortfolioState) -> dict[str, float]:
        weights: dict[str, float] = {}
        for d in decisions:
            if abs(d.target_position) <= self.eps:
                continue
            edge = 2.0 * d.confidence - 1.0
            if edge <= 0:
                continue
            size = self.fraction * edge
            weights[d.asset] = weights.get(d.asset, 0.0) + float(np.sign(d.target_position) * size)
        return _normalise_to_unit_gross(weights)


class RegimeAware(Allocator):
    """Per-regime weight overrides."""

    def __init__(
        self,
        regime_weights: dict[str, dict[str, float]],
        fallback: Allocator | None = None,
    ) -> None:
        self.regime_weights = {
            r: {a: float(w) for a, w in m.items()}
            for r, m in regime_weights.items()
        }
        self.fallback = fallback or EqualWeight()
        self.name = "regime_aware"

    def allocate(self, decisions: list[Decision], portfolio: PortfolioState) -> dict[str, float]:
        regime = portfolio.regime
        if regime is None or regime not in self.regime_weights:
            return self.fallback.allocate(decisions, portfolio)
        override = self.regime_weights[regime]
        sign_by_asset = {
            d.asset: (1.0 if d.target_position > 0 else (-1.0 if d.target_position < 0 else 0.0))
            for d in decisions
        }
        weights: dict[str, float] = {}
        for asset, w in override.items():
            sign = sign_by_asset.get(asset, 1.0 if w >= 0 else -1.0)
            if sign == 0.0:
                continue
            weights[asset] = float(sign * abs(w))
        if not weights:
            return self.fallback.allocate(decisions, portfolio)
        return _normalise_to_unit_gross(weights)


__all__ = ["Allocator", "EqualWeight", "RiskParity", "KellyOnline", "RegimeAware"]
