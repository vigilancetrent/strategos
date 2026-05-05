"""Capital allocators."""
from strategos.allocation.allocators import (
    Allocator,
    EqualWeight,
    RiskParity,
    KellyOnline,
    RegimeAware,
)

__all__ = ["Allocator", "EqualWeight", "RiskParity", "KellyOnline", "RegimeAware"]
