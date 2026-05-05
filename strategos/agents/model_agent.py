"""Model agent wrapper — plug any sklearn-style ``predict`` model."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from strategos.agents.base import Agent
from strategos.types import Decision, PortfolioState


class ModelAgent(Agent):
    """Wrap any sklearn-style estimator (must expose ``predict``)."""

    def __init__(
        self,
        name: str,
        model: Any,
        feature_extractor: Callable[[dict, PortfolioState], Any],
        predict_to_decision: Callable[[Any, str], Decision],
        use_proba: bool = False,
    ) -> None:
        self.name = name
        self._model = model
        self._features = feature_extractor
        self._predict_to_decision = predict_to_decision
        self._use_proba = use_proba

    def act(self, obs: dict, portfolio: PortfolioState) -> Decision:
        X = self._features(obs, portfolio)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._use_proba and hasattr(self._model, "predict_proba"):
            pred = self._model.predict_proba(X)
        elif hasattr(self._model, "predict"):
            pred = self._model.predict(X)
        else:
            raise TypeError(
                f"ModelAgent {self.name!r}: model must expose .predict()"
            )
        decision = self._predict_to_decision(pred, self.name)
        if not isinstance(decision, Decision):
            raise TypeError(
                f"ModelAgent {self.name!r}: predict_to_decision must return Decision"
            )
        return decision


__all__ = ["ModelAgent"]
