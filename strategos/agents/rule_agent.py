"""Rule-based agent — wraps an arbitrary callable into the Agent interface."""
from __future__ import annotations

from typing import Callable, Tuple, Union

from strategos.agents.base import Agent
from strategos.types import Decision, PortfolioState

RuleFn = Callable[[dict, PortfolioState], Union[Decision, Tuple[str, float, float]]]


class RuleAgent(Agent):
    """Wrap any callable ``fn(obs, portfolio)`` into an :class:`Agent`."""

    def __init__(self, name: str, fn: RuleFn, default_asset: str | None = None) -> None:
        self.name = name
        self._fn = fn
        self._default_asset = default_asset

    def act(self, obs: dict, portfolio: PortfolioState) -> Decision:
        out = self._fn(obs, portfolio)
        if isinstance(out, Decision):
            if out.agent_name != self.name:
                return Decision(
                    agent_name=self.name,
                    asset=out.asset,
                    target_position=out.target_position,
                    confidence=out.confidence,
                    reasoning=out.reasoning,
                    metadata=dict(out.metadata),
                )
            return out
        if isinstance(out, tuple) and len(out) == 3:
            asset, pos, conf = out
            return Decision(
                agent_name=self.name,
                asset=str(asset),
                target_position=float(pos),
                confidence=float(conf),
                reasoning=f"rule:{self.name}",
            )
        raise TypeError(
            f"RuleAgent {self.name!r} returned unsupported type {type(out).__name__}; "
            "expected Decision or (asset, target_position, confidence)"
        )


__all__ = ["RuleAgent"]
