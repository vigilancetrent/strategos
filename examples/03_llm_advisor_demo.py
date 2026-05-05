"""Demonstrate the strategos LLM advisor end-to-end.

Run:
    python examples/03_llm_advisor_demo.py

Configuration (env vars):
    STRATEGOS_LLM_PROVIDER  anthropic | openai | ollama   (auto-detected if unset)
    STRATEGOS_LLM_MODEL     optional model override
    ANTHROPIC_API_KEY       required for anthropic
    OPENAI_API_KEY          required for openai
    OLLAMA_HOST             default http://localhost:11434
    OLLAMA_MODEL            default llama3.1

If no provider can be initialised, the demo prints a clear instructional
message and exits 0 — it never crashes.
"""

from __future__ import annotations

import sys
import textwrap

from strategos.llm import LLMAdvisor, from_env


SYNTHETIC_AUDIT = [
    {
        "ts": "2026-05-05T14:00:00Z",
        "agent": "momentum_v2",
        "signal": 0.73,
        "decision": "BUY",
        "size_pct": 0.08,
        "regime": "trending_up",
    },
    {
        "ts": "2026-05-05T14:00:01Z",
        "gate": "concentration",
        "verdict": "PASS",
        "post_weight": 0.31,
    },
    {
        "ts": "2026-05-05T14:00:01Z",
        "gate": "drawdown",
        "verdict": "PASS",
        "headroom_pct": 0.41,
    },
    {
        "ts": "2026-05-05T14:00:02Z",
        "execution": "filled",
        "slippage_bps": 4.2,
    },
]

SYNTHETIC_DECISIONS = [
    {"agent": "momentum_v2", "instrument": "BTCUSD", "size_pct": 0.08, "side": "BUY"},
    {"agent": "meanrev_v1", "instrument": "ETHUSD", "size_pct": 0.04, "side": "SELL"},
]

SYNTHETIC_PORTFOLIO = {
    "cash_pct": 0.55,
    "positions": {"BTCUSD": 0.31, "ETHUSD": -0.04, "SOLUSD": 0.10},
    "leverage": 1.0,
    "drawdown_pct": -0.018,
}

SYNTHETIC_INDICATORS = {
    "adx": 32.7,
    "realized_vol_30d": 0.41,
    "vol_z": -0.6,
    "trend_strength": 0.78,
    "regime_persistence": 0.91,
}


def _banner(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def main() -> int:
    try:
        provider = from_env()
    except RuntimeError as exc:
        print(
            textwrap.dedent(
                f"""\
                strategos LLM advisor demo could not initialise a provider.

                Reason: {exc}

                To run this demo, set ONE of the following:

                  - ANTHROPIC_API_KEY=sk-...        (Anthropic Claude)
                  - OPENAI_API_KEY=sk-...           (OpenAI)
                  - A running Ollama daemon at $OLLAMA_HOST (default
                    http://localhost:11434), e.g.:
                        ollama serve
                        ollama pull llama3.1

                Optional: STRATEGOS_LLM_PROVIDER=anthropic|openai|ollama
                         STRATEGOS_LLM_MODEL=<model-id>
                """
            )
        )
        return 0

    print(f"Using provider: {provider.name} (default model: {provider.default_model})")
    advisor = LLMAdvisor(provider)

    _banner("strategy_brief")
    print(
        advisor.strategy_brief(
            agents=["momentum_v2", "meanrev_v1", "carry_v3"],
            regime="trending_up",
            recent_pnl=0.0241,
        )
    )

    _banner("regime_explain")
    print(advisor.regime_explain(regime="trending_up", indicators=SYNTHETIC_INDICATORS))

    _banner("risk_review")
    print(advisor.risk_review(decisions=SYNTHETIC_DECISIONS, portfolio=SYNTHETIC_PORTFOLIO))

    _banner("post_mortem")
    print(advisor.post_mortem(SYNTHETIC_AUDIT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
