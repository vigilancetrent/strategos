"""strategos — agentic trading orchestrator.

Top-level re-exports. The optional ``strategos.llm`` package is *not*
imported here — import it explicitly if you want LLM advisor support.
"""

from strategos.types import (
    Decision,
    Verdict,
    PortfolioState,
    AuditEvent,
)
from strategos.agents import (
    Agent,
    RuleAgent,
    RLAgent,
    ModelAgent,
)
from strategos.risk import (
    RiskGate,
    PositionLimit,
    ExposureCap,
    DrawdownStop,
    VaRLimit,
    KillSwitch,
    RegimeGate,
    compose,
)
from strategos.allocation import (
    Allocator,
    EqualWeight,
    RiskParity,
    KellyOnline,
    RegimeAware,
)
from strategos.ensemble import (
    Ensemble,
    PerformanceWeighted,
    RegimeConditioned,
    MajorityVote,
)
from strategos.audit import AuditLog
from strategos.orchestrator import Orchestrator

__version__ = "0.1.0"

__all__ = [
    "Decision",
    "Verdict",
    "PortfolioState",
    "AuditEvent",
    "Agent",
    "RuleAgent",
    "RLAgent",
    "ModelAgent",
    "RiskGate",
    "PositionLimit",
    "ExposureCap",
    "DrawdownStop",
    "VaRLimit",
    "KillSwitch",
    "RegimeGate",
    "compose",
    "Allocator",
    "EqualWeight",
    "RiskParity",
    "KellyOnline",
    "RegimeAware",
    "Ensemble",
    "PerformanceWeighted",
    "RegimeConditioned",
    "MajorityVote",
    "AuditLog",
    "Orchestrator",
    "__version__",
]
