"""Ensembles — collapse multiple agent decisions per asset into one."""
from strategos.ensemble.ensemble import (
    Ensemble,
    PerformanceWeighted,
    RegimeConditioned,
    MajorityVote,
)

__all__ = ["Ensemble", "PerformanceWeighted", "RegimeConditioned", "MajorityVote"]
