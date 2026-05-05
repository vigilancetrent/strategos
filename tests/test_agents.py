"""Smoke tests for the agent layer."""
from __future__ import annotations

import numpy as np
import pytest

from strategos import Decision, PortfolioState, RuleAgent, RLAgent, ModelAgent


def _pf() -> PortfolioState:
    return PortfolioState(cash=100_000.0, positions={"AAPL": 0.0}, last_prices={"AAPL": 100.0})


def test_rule_agent_tuple_return():
    agent = RuleAgent(name="trend", fn=lambda obs, pf: ("AAPL", 0.5, 0.8))
    d = agent.act({"x": 1}, _pf())
    assert isinstance(d, Decision)
    assert d.agent_name == "trend"
    assert d.asset == "AAPL"
    assert d.target_position == 0.5
    assert d.confidence == 0.8


def test_rule_agent_decision_return_overrides_name():
    def fn(obs, pf):
        return Decision(
            agent_name="liar",
            asset="AAPL",
            target_position=-0.4,
            confidence=0.6,
        )
    agent = RuleAgent(name="actual", fn=fn)
    d = agent.act({}, _pf())
    assert d.agent_name == "actual"
    assert d.target_position == -0.4


def test_rule_agent_bad_return_raises():
    agent = RuleAgent(name="bad", fn=lambda obs, pf: 42)
    with pytest.raises(TypeError):
        agent.act({}, _pf())


def test_decision_validation():
    with pytest.raises(ValueError):
        Decision(agent_name="x", asset="A", target_position=1.5, confidence=0.5)
    with pytest.raises(ValueError):
        Decision(agent_name="x", asset="A", target_position=0.0, confidence=2.0)


def test_rl_agent_with_predict():
    class Policy:
        def predict(self, x, deterministic=True):
            return np.array([0.3]), None

    def obs_to_array(obs, pf):
        return np.array([obs["x"]])

    def action_to_decision(action, name):
        return Decision(agent_name=name, asset="AAPL", target_position=float(action[0]), confidence=0.7)

    agent = RLAgent(name="rl", policy=Policy(), obs_to_array=obs_to_array, action_to_decision=action_to_decision)
    d = agent.act({"x": 1.0}, _pf())
    assert d.target_position == pytest.approx(0.3)
    assert d.agent_name == "rl"


def test_rl_agent_with_callable_policy():
    def policy(x):
        return np.array([-0.2])

    def obs_to_array(obs, pf):
        return np.array([obs["x"]])

    def action_to_decision(action, name):
        return Decision(agent_name=name, asset="AAPL", target_position=float(action[0]), confidence=0.6)

    agent = RLAgent(name="rl_cb", policy=policy, obs_to_array=obs_to_array, action_to_decision=action_to_decision)
    d = agent.act({"x": 0.0}, _pf())
    assert d.target_position == pytest.approx(-0.2)


def test_model_agent_basic():
    class M:
        def predict(self, X):
            return np.array([0.4 * X[0, 0]])

    def features(obs, pf):
        return [obs["x"]]

    def to_decision(pred, name):
        return Decision(agent_name=name, asset="AAPL", target_position=float(pred[0]), confidence=0.55)

    agent = ModelAgent(name="ml", model=M(), feature_extractor=features, predict_to_decision=to_decision)
    d = agent.act({"x": 1.0}, _pf())
    assert d.target_position == pytest.approx(0.4)
