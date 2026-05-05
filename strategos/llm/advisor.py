"""High-level LLM advisor used by the strategos orchestrator.

The advisor uses an identical system prompt across every method so that
providers with prompt caching (Anthropic ephemeral; OpenAI auto) only pay
for it once and cheaply re-use it for every subsequent call within the
cache window.

Each method returns a plain ``str`` and never raises: provider errors are
swallowed and surfaced as ``"[advisor offline: ...]"``.
"""

from __future__ import annotations

import json
from typing import Any

from strategos.llm.base import BaseLLMProvider, Message


_SYSTEM_PROMPT = (
    "You are a senior quantitative risk advisor reviewing automated trading "
    "decisions made by a multi-agent orchestrator named 'strategos'. Your "
    "role is strictly advisory and post-hoc.\n"
    "\n"
    "Rules:\n"
    "1. Be terse and factual. Prefer 3-6 short sentences over long prose.\n"
    "2. Cite the specific decision, agent, regime, or indicator when relevant.\n"
    "3. NEVER recommend new trades or position sizes. You review what was "
    "decided, you do not decide.\n"
    "4. Flag risks the automated gates may have missed (concentration, "
    "correlation, liquidity, regime mismatch, model staleness).\n"
    "5. If the inputs are insufficient, say so explicitly rather than "
    "speculating.\n"
    "6. Use plain English. No emojis, no marketing language, no hedging "
    "like 'it depends'.\n"
)


class LLMAdvisor:
    """Wrap a :class:`BaseLLMProvider` with strategos-specific helpers."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    def strategy_brief(
        self,
        agents: list[str],
        regime: str | None,
        recent_pnl: float,
    ) -> str:
        """Pre-trade brief on the recommended posture for the next interval."""
        regime_str = regime or "unknown"
        agents_str = ", ".join(agents) if agents else "(none registered)"
        user = (
            "PRE-TRADE BRIEF REQUEST\n"
            f"- Regime: {regime_str}\n"
            f"- Active agents: {agents_str}\n"
            f"- Trailing PnL: {recent_pnl:+.4f}\n"
            "\n"
            "Write a concise pre-trade posture brief (3-5 sentences). "
            "Highlight which agent profiles are best suited to this regime "
            "and which carry elevated risk. Do not propose specific trades."
        )
        return self._call(user, max_tokens=400)

    def post_mortem(self, audit_events: list[dict]) -> str:
        """Narrative post-mortem of the most recent decision audit trail."""
        if not audit_events:
            return "[advisor offline: no audit events provided]"

        trimmed = audit_events[-50:]
        payload = _safe_json(trimmed, max_chars=6000)
        user = (
            "POST-MORTEM REQUEST\n"
            "Below is the audit trail of recent orchestrator events "
            "(JSON list). Write a short post-mortem (4-7 sentences) "
            "explaining what was decided, why, and what — if anything — "
            "should be reviewed.\n"
            "\n"
            "AUDIT_TRAIL:\n"
            f"{payload}"
        )
        return self._call(user, max_tokens=600)

    def risk_review(
        self,
        decisions: list[dict],
        portfolio: dict,
    ) -> str:
        """Independent review surfacing risks the gates may have missed."""
        decisions_payload = _safe_json(decisions[-20:], max_chars=4000)
        portfolio_payload = _safe_json(portfolio, max_chars=2000)
        user = (
            "RISK REVIEW REQUEST\n"
            "Review the recent decisions and current portfolio state below. "
            "Identify up to 3 specific risk concerns that the automated "
            "risk gates may have missed (concentration, correlation, "
            "liquidity, regime mismatch, stale signals, sizing drift). "
            "Format as a short bullet list. If you see no concerns, say so "
            "in one sentence.\n"
            "\n"
            "DECISIONS:\n"
            f"{decisions_payload}\n"
            "\n"
            "PORTFOLIO:\n"
            f"{portfolio_payload}"
        )
        return self._call(user, max_tokens=500)

    def regime_explain(self, regime: str, indicators: dict) -> str:
        """One-paragraph plain-English explanation of the current regime."""
        ind_payload = _safe_json(indicators, max_chars=1500)
        user = (
            "REGIME EXPLANATION REQUEST\n"
            f"Current regime label: {regime}\n"
            "Supporting indicators:\n"
            f"{ind_payload}\n"
            "\n"
            "In one short paragraph (3-4 sentences) explain in plain "
            "English why the inputs above support this regime label. Cite "
            "the specific indicator values you relied on."
        )
        return self._call(user, max_tokens=300)

    def _call(self, user_prompt: str, *, max_tokens: int) -> str:
        try:
            resp = self._provider.chat(
                [Message(role="user", content=user_prompt)],
                system=_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=0.2,
                cache_system=True,
            )
        except Exception as exc:  # noqa: BLE001
            return f"[advisor offline: {type(exc).__name__}: {exc}]"

        text = (resp.text or "").strip()
        if not text:
            return "[advisor offline: empty response from provider]"
        return text


def _safe_json(obj: Any, *, max_chars: int) -> str:
    """JSON-serialise ``obj`` defensively, truncating if too long."""
    try:
        s = json.dumps(obj, default=str, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        s = repr(obj)
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n... [truncated, original length {len(s)} chars]"
    return s
