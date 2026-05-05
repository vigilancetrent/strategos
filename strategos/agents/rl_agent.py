"""RL agent wrapper — plug a Gym-style policy or stable-baselines3 model."""
from __future__ import annotations

from typing import Any, Callable

from strategos.agents.base import Agent
from strategos.types import Decision, PortfolioState


class RLAgent(Agent):
    """Wrap a reinforcement-learning policy into an :class:`Agent`."""

    def __init__(
        self,
        name: str,
        policy: Any,
        obs_to_array: Callable[[dict, PortfolioState], Any],
        action_to_decision: Callable[[Any, str], Decision],
        deterministic: bool = True,
    ) -> None:
        self.name = name
        self._policy = policy
        self._obs_to_array = obs_to_array
        self._action_to_decision = action_to_decision
        self._deterministic = deterministic

    def act(self, obs: dict, portfolio: PortfolioState) -> Decision:
        x = self._obs_to_array(obs, portfolio)
        if hasattr(self._policy, "predict"):
            try:
                action, _state = self._policy.predict(x, deterministic=self._deterministic)
            except TypeError:
                action, _state = self._policy.predict(x)
        elif callable(self._policy):
            action = self._policy(x)
        else:
            raise TypeError(
                f"RLAgent {self.name!r}: policy must be callable or expose .predict()"
            )
        decision = self._action_to_decision(action, self.name)
        if not isinstance(decision, Decision):
            raise TypeError(
                f"RLAgent {self.name!r}: action_to_decision must return a Decision, "
                f"got {type(decision).__name__}"
            )
        return decision


__all__ = ["RLAgent"]
