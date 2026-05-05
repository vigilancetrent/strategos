"""The orchestrator main loop."""
from __future__ import annotations

from typing import Any

from strategos.agents.base import Agent
from strategos.allocation.allocators import Allocator
from strategos.audit.log import AuditLog
from strategos.ensemble.ensemble import Ensemble
from strategos.risk.gates import RiskGate, compose
from strategos.types import Decision, PortfolioState


class Orchestrator:
    """The boss layer — agents -> ensemble -> alloc -> gates -> audit."""

    def __init__(
        self,
        agents: list[Agent],
        ensemble: Ensemble,
        allocator: Allocator,
        risk_gates: list[RiskGate],
        audit_log: AuditLog | None = None,
        llm_advisor: Any | None = None,
    ) -> None:
        if not agents:
            raise ValueError("Orchestrator requires at least one agent")
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate agent names: {names}")
        self.agents = list(agents)
        self.ensemble = ensemble
        self.allocator = allocator
        self.risk_gates = list(risk_gates)
        self.audit_log = audit_log if audit_log is not None else AuditLog()
        self.llm_advisor = llm_advisor

    def step(self, obs: dict, portfolio: PortfolioState) -> dict[str, float]:
        step_events: list[dict] = []

        def _record(kind: str, payload: dict) -> None:
            evt = self.audit_log.append(kind, payload)
            step_events.append(
                {"seq": evt.seq, "kind": evt.kind, "payload": evt.payload}
            )

        decisions: list[Decision] = []
        for agent in self.agents:
            try:
                decision = agent.act(obs, portfolio)
            except Exception as exc:
                _record(
                    "skip",
                    {
                        "stage": "agent_act",
                        "agent": agent.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            if not isinstance(decision, Decision):
                _record(
                    "skip",
                    {
                        "stage": "agent_act",
                        "agent": agent.name,
                        "error": "agent did not return a Decision",
                    },
                )
                continue
            decisions.append(decision)
            _record(
                "decision",
                {
                    "agent": decision.agent_name,
                    "asset": decision.asset,
                    "target_position": decision.target_position,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "metadata": decision.metadata,
                },
            )

        if not decisions:
            _record("execution", {"weights": {}, "note": "no_decisions"})
            self._dispatch_advisor(step_events)
            return {}

        try:
            combined = self.ensemble.combine(decisions, portfolio)
        except Exception as exc:
            _record("skip", {"stage": "ensemble", "error": f"{type(exc).__name__}: {exc}"})
            self._dispatch_advisor(step_events)
            return {}

        _record(
            "ensemble",
            {
                "n_in": len(decisions),
                "n_out": len(combined),
                "decisions": [
                    {
                        "agent": d.agent_name,
                        "asset": d.asset,
                        "target_position": d.target_position,
                        "confidence": d.confidence,
                    }
                    for d in combined
                ],
            },
        )

        try:
            weights = self.allocator.allocate(combined, portfolio)
        except Exception as exc:
            _record("skip", {"stage": "allocate", "error": f"{type(exc).__name__}: {exc}"})
            self._dispatch_advisor(step_events)
            return {}

        _record("allocate", {"weights": weights, "allocator": getattr(self.allocator, "name", "?")})

        gate_chain = compose(self.risk_gates) if self.risk_gates else None
        final_weights: dict[str, float] = {}
        for asset, w in weights.items():
            try:
                candidate = Decision(
                    agent_name="orchestrator",
                    asset=asset,
                    target_position=float(max(-1.0, min(1.0, w))),
                    confidence=1.0,
                    reasoning="post_allocation_candidate",
                )
            except ValueError as exc:
                _record(
                    "skip",
                    {"stage": "candidate", "asset": asset, "weight": w, "error": str(exc)},
                )
                final_weights[asset] = 0.0
                continue
            if gate_chain is None:
                final_weights[asset] = candidate.target_position
                _record(
                    "verdict",
                    {"asset": asset, "ok": True, "reason": "no_gates", "adjusted": candidate.target_position},
                )
                continue
            verdict = gate_chain.review(candidate, portfolio)
            _record(
                "verdict",
                {
                    "asset": asset,
                    "ok": verdict.ok,
                    "reason": verdict.reason,
                    "adjusted": verdict.adjusted_position,
                    "candidate": candidate.target_position,
                },
            )
            final_weights[asset] = (
                float(verdict.adjusted_position)
                if verdict.ok and verdict.adjusted_position is not None
                else 0.0
            )

        _record("execution", {"weights": final_weights})
        self._dispatch_advisor(step_events)

        return final_weights

    def _dispatch_advisor(self, step_events: list[dict]) -> None:
        """Call ``llm_advisor.post_mortem`` if set; never raise."""
        if self.llm_advisor is None:
            return
        try:
            post_mortem = getattr(self.llm_advisor, "post_mortem", None)
            if post_mortem is None:
                return
            post_mortem(step_events)
        except Exception as exc:
            try:
                self.audit_log.append(
                    "system",
                    {
                        "stage": "llm_advisor",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            except Exception:
                pass


__all__ = ["Orchestrator"]
