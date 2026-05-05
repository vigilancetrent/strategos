"""Agent abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

from strategos.types import Decision, PortfolioState


class Agent(ABC):
    """Base class for any signal-producing agent."""

    name: str = "agent"

    @abstractmethod
    def act(self, obs: dict, portfolio: PortfolioState) -> Decision:
        """Return a single :class:`Decision`."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} name={self.name!r}>"


__all__ = ["Agent"]
