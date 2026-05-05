# strategos

**The boss layer for the quantflow ecosystem.**
Multi-agent ensemble · composable risk gates · regime-aware allocation · regulator-ready audit log · optional LLM advisor (Anthropic / OpenAI / Ollama).

`strategos` is the agentic trading orchestrator. It pulls signals from N agents (rule-based, RL, ML), runs them through composable risk middleware, allocates capital regime-aware, and writes a tamper-evident audit log of every decision. Think of it as the supervisor layer that says **yes / no / size-at-X** for every trade — and can defend that decision in front of a regulator.

```
            +----------+   +----------+   +----------+
   obs ---> | Agent A  |   | Agent B  |   | Agent N  |
            +----+-----+   +----+-----+   +----+-----+
                 |              |              |
                 v              v              v
            +-------------------------------------+
            |             Ensemble                |  (perf-weighted / regime / vote)
            +-------------------+-----------------+
                                |
                                v
            +-------------------------------------+
            |             Allocator               |  (equal / risk-parity / Kelly / regime-aware)
            +-------------------+-----------------+
                                |
                                v
            +-------------------------------------+
            |     Risk gates (composed chain)     |  (limit -> exposure -> DD -> VaR -> kill -> regime)
            +-------------------+-----------------+
                                |
                                v
                       target_weights
                                |
            +-------------------+-----------------+
            |       Audit log (HMAC chain)        | <-- LLM advisor (optional)
            +-------------------------------------+
```

## Why strategos

| Feature                          | strategos | FinRL  | qlib   | hand-rolled |
|----------------------------------|:---------:|:------:|:------:|:-----------:|
| Multi-agent ensemble             | yes       | no     | partial| maybe       |
| Composable risk middleware       | yes       | no     | no     | no          |
| Regime-aware allocation          | yes       | no     | no     | no          |
| Tamper-evident audit log         | yes       | no     | no     | no          |
| LLM post-mortem advisor          | yes       | no     | no     | no          |
| Pluggable agents (rule/RL/ML)    | yes       | RL only| ML only| yes         |
| Regulator-ready by design        | yes       | no     | no     | no          |
| Zero forced framework            | yes       | no     | no     | yes         |

## Quickstart

```python
import numpy as np
from strategos import (
    Orchestrator, RuleAgent, EqualWeight, MajorityVote,
    PositionLimit, ExposureCap, DrawdownStop, AuditLog, PortfolioState,
)

trend = RuleAgent(
    name="trend",
    fn=lambda obs, pf: ("AAPL", 0.6 if obs["ret_5"] > 0 else -0.3, 0.7),
)
mr = RuleAgent(
    name="mean_revert",
    fn=lambda obs, pf: ("AAPL", -0.5 if obs["ret_1"] > 0.02 else 0.2, 0.5),
)

orc = Orchestrator(
    agents=[trend, mr],
    ensemble=MajorityVote(),
    allocator=EqualWeight(),
    risk_gates=[PositionLimit(0.5), ExposureCap(1.0), DrawdownStop(0.2)],
    audit_log=AuditLog(path="audit.jsonl", secret_key=b"my-secret-key" * 4),
)

pf = PortfolioState(cash=100_000.0, positions={"AAPL": 0.0}, last_prices={"AAPL": 190.0})
weights = orc.step({"ret_1": 0.005, "ret_5": 0.02}, pf)
print(weights)  # {"AAPL": 0.5}
```

## With an LLM advisor

```python
from strategos.llm import from_env, LLMAdvisor

provider = from_env()              # picks anthropic | openai | ollama from env
advisor = LLMAdvisor(provider)
orc = Orchestrator(..., llm_advisor=advisor)
# advisor.post_mortem() runs after every step — fire-and-forget
```

## Module overview

| Module                       | Purpose                                                  |
|------------------------------|----------------------------------------------------------|
| `strategos.types`            | `Decision`, `Verdict`, `PortfolioState`, `AuditEvent`    |
| `strategos.agents`           | `Agent` ABC + rule/RL/ML wrappers                        |
| `strategos.risk`             | Composable risk gates (limit, exposure, DD, VaR, kill, regime) |
| `strategos.allocation`       | EqualWeight, RiskParity, KellyOnline, RegimeAware        |
| `strategos.ensemble`         | PerformanceWeighted, RegimeConditioned, MajorityVote     |
| `strategos.audit`            | HMAC-chained append-only JSON-line log                   |
| `strategos.orchestrator`     | The main loop: agents -> ensemble -> alloc -> gates      |
| `strategos.llm` *(optional)* | Post-mortem LLM advisor (Anthropic / OpenAI / Ollama)    |

## Sibling repos

`strategos` is one piece of the **quantflow** ecosystem by [vigilancetrent](https://github.com/vigilancetrent):

- [`quantflow`](https://github.com/vigilancetrent/quantflow) — backtesting engine
- [`synthflow`](https://github.com/vigilancetrent/synthflow) — synthetic market data generator
- [`regimecast`](https://github.com/vigilancetrent/regimecast) — market regime detection (feed `portfolio.regime`)
- [`signalstack`](https://github.com/vigilancetrent/signalstack) — live cyberpunk dashboard

`strategos` deliberately does **not** import these at module top level — bring whichever you want.

## Install

```bash
pip install strategos
pip install "strategos[all-llm]"   # with Anthropic + OpenAI + Ollama clients
pip install "strategos[dev]"        # with pytest
```

## Roadmap

- [ ] Online learning loops (agents update from realised PnL)
- [ ] First-class multi-asset / cross-sectional allocation
- [ ] Broker adapters (IBKR, Alpaca, CCXT) — read-only first
- [ ] CRDT-based distributed audit ledger across nodes
- [ ] Streaming orchestration (async `step`)
- [ ] Replay tooling: re-run an audit log against a new policy

## License

MIT — `(c) 2026 thechifura and strategos contributors`. See `LICENSE`.
