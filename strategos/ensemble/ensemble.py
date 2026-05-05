"""Ensemble voters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Deque

import numpy as np

from strategos.types import Decision, PortfolioState


class Ensemble(ABC):
    """Collapse multiple agent decisions per asset into one."""

    name: str = "ensemble"

    @abstractmethod
    def combine(
        self,
        decisions: list[Decision],
        portfolio: PortfolioState,
    ) -> list[Decision]:
        raise NotImplementedError


def _group_by_asset(decisions: list[Decision]) -> dict[str, list[Decision]]:
    g: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        g[d.asset].append(d)
    return g


def _clip_position(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


class PerformanceWeighted(Ensemble):
    """Weight agents by their trailing realised PnL."""

    def __init__(self, window: int = 50, temperature: float = 1.0) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.window = int(window)
        self.temperature = float(temperature)
        self._pnl: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        self.name = "performance_weighted"

    def record_pnl(self, agent_name: str, pnl: float) -> None:
        self._pnl[agent_name].append(float(pnl))

    def _weights_for(self, agent_names: list[str]) -> dict[str, float]:
        scores = np.array(
            [sum(self._pnl[name]) if self._pnl[name] else 0.0 for name in agent_names],
            dtype=float,
        )
        if not np.any(scores):
            return {name: 1.0 / len(agent_names) for name in agent_names}
        scaled = scores / self.temperature
        scaled -= scaled.max()
        exps = np.exp(scaled)
        denom = exps.sum()
        if denom <= 0 or not np.isfinite(denom):
            return {name: 1.0 / len(agent_names) for name in agent_names}
        ws = exps / denom
        return {name: float(w) for name, w in zip(agent_names, ws)}

    def combine(self, decisions: list[Decision], portfolio: PortfolioState) -> list[Decision]:
        out: list[Decision] = []
        for asset, group in _group_by_asset(decisions).items():
            agent_names = [d.agent_name for d in group]
            weights = self._weights_for(agent_names)
            target = sum(weights[d.agent_name] * d.target_position for d in group)
            conf = float(np.mean([d.confidence for d in group]))
            out.append(
                Decision(
                    agent_name="ensemble:performance_weighted",
                    asset=asset,
                    target_position=_clip_position(target),
                    confidence=conf,
                    reasoning=f"weighted_mean over {len(group)} agents",
                    metadata={"weights": weights},
                )
            )
        return out


class RegimeConditioned(Ensemble):
    """Per-regime agent weights."""

    def __init__(self, regime_to_weights: dict[str, dict[str, float]]) -> None:
        self.regime_to_weights = {
            r: {n: float(w) for n, w in m.items()} for r, m in regime_to_weights.items()
        }
        self.name = "regime_conditioned"

    def _resolve_weights(self, regime: str | None, agent_names: list[str]) -> dict[str, float]:
        if regime is None or regime not in self.regime_to_weights:
            return {name: 1.0 / len(agent_names) for name in agent_names}
        raw = self.regime_to_weights[regime]
        present = {name: max(0.0, raw.get(name, 0.0)) for name in agent_names}
        total = sum(present.values())
        if total <= 0:
            return {name: 1.0 / len(agent_names) for name in agent_names}
        return {name: w / total for name, w in present.items()}

    def combine(self, decisions: list[Decision], portfolio: PortfolioState) -> list[Decision]:
        regime = portfolio.regime
        out: list[Decision] = []
        for asset, group in _group_by_asset(decisions).items():
            agent_names = [d.agent_name for d in group]
            weights = self._resolve_weights(regime, agent_names)
            target = sum(weights[d.agent_name] * d.target_position for d in group)
            conf = float(np.mean([d.confidence for d in group]))
            out.append(
                Decision(
                    agent_name="ensemble:regime_conditioned",
                    asset=asset,
                    target_position=_clip_position(target),
                    confidence=conf,
                    reasoning=f"regime={regime} weights={weights}",
                    metadata={"weights": weights, "regime": regime},
                )
            )
        return out


class MajorityVote(Ensemble):
    """Confidence-weighted sign vote."""

    def __init__(self) -> None:
        self.name = "majority_vote"

    def combine(self, decisions: list[Decision], portfolio: PortfolioState) -> list[Decision]:
        out: list[Decision] = []
        for asset, group in _group_by_asset(decisions).items():
            confs = np.array([d.confidence for d in group], dtype=float)
            tgts = np.array([d.target_position for d in group], dtype=float)
            denom = confs.sum()
            if denom <= 0:
                target = 0.0
                conf = 0.0
            else:
                signed_score = float(np.sum(confs * tgts))
                magnitude = float(np.sum(confs * np.abs(tgts)) / denom)
                sign = 1.0 if signed_score > 0 else (-1.0 if signed_score < 0 else 0.0)
                target = sign * magnitude
                conf = float(denom / len(group))
            out.append(
                Decision(
                    agent_name="ensemble:majority_vote",
                    asset=asset,
                    target_position=_clip_position(target),
                    confidence=float(np.clip(conf, 0.0, 1.0)),
                    reasoning=f"vote over {len(group)} agents",
                )
            )
        return out


__all__ = ["Ensemble", "PerformanceWeighted", "RegimeConditioned", "MajorityVote"]
