# Forex Trading Bot — Capital.com

A modular, low/medium-frequency Forex **CFD** trading bot for the
[Capital.com](https://capital.com) API. It covers the full pipeline described in
the project plan: historical data → backtesting → demo trading → cautious live
trading, with strong risk controls and a clean separation between strategy logic
and infrastructure.

> **Risk warning.** Trading CFDs on margin carries a high risk of losing money.
> This software is provided for research and education. Always validate on a
> **demo** account first, use conservative sizing, and never risk capital you
> cannot afford to lose.

## Design principles

- **Strategies are pure decision functions.** A strategy receives a closed
  candle plus a read-only context and emits an *intent* (`Signal`). It never
  touches the broker, places orders, or persists state. The exact same strategy
  code runs in backtest and live.
- **The core is pure-stdlib.** Models, indicators, strategies, risk, the
  backtester and metrics have **no third-party dependencies**, so they run and
  test anywhere. Heavier libraries (`requests`, `websocket-client`, `pandas`,
  `matplotlib`) are only needed for live connectivity / richer analytics and are
  imported lazily.
- **One risk + execution path.** The backtester and the live engine share the
  same `RiskManager`, so position sizing and loss/exposure limits are validated
  identically offline and online.

## Architecture

```
forex_bot/
├── models.py            Typed domain models: Candle, Signal, Order, Position, Trade
├── config.py            Credentials (.env) + trading/risk config (YAML)
├── logging_setup.py     Structured key=value logging
├── indicators.py        Pure indicators: SMA, EMA, RSI, ATR, True Range
├── strategy/            Pluggable strategies (registry-based)
│   ├── base.py          StrategyBase + StrategyContext
│   ├── ema_crossover.py EMA crossover trend-follower (ATR stops/targets)
│   └── rsi_reversion.py RSI mean-reversion with neutral-band exits
├── risk/manager.py      Fixed-fractional sizing + portfolio limits + daily halt
├── execution/           Order → Fill
│   ├── simulated.py     Spread/slippage/commission model for backtests
│   └── live.py          Capital.com order placement + deal confirmation
├── backtest/            Candle-replay engine
│   ├── engine.py        Strategy→risk→execution replay (no look-ahead)
│   ├── portfolio.py     Cash/positions/trades/equity-curve tracking
│   ├── metrics.py       Return, CAGR, Sharpe/Sortino, drawdown, profit factor…
│   └── reporting.py     CSV + HTML report export
├── api/                 Capital.com clients
│   ├── rest_client.py   Session mgmt, history, accounts, positions, orders
│   ├── websocket_client.py  Live quote streaming (≤40 instruments/session)
│   └── rate_limiter.py  Request throttling
├── data/                Data layer
│   ├── storage.py       CSV historical store + data-quality checks
│   └── candle_builder.py  Aggregate live quotes into timeframe candles
├── live_engine.py       Wires quotes→candles→strategy→risk→live execution
└── cli.py               download / backtest / demo / live
```

These map onto the plan's phases:

| Plan phase | Where |
|---|---|
| Phase 1 — API access (REST + WebSocket) | `api/` |
| Phase 2 — Data layer & backtester | `data/`, `backtest/` |
| Phase 3 — Strategy research & interface | `strategy/`, `indicators.py` |
| Phase 4 — Live engine (demo → live) | `live_engine.py`, `execution/live.py` |
| Phase 5 — Logging / risk / ops | `logging_setup.py`, `risk/` |

## Quick start (no broker account needed)

The backtesting path runs on the standard library alone.

```bash
# 1. Create a config from the example
cp config/config.example.yaml config/config.yaml

# 2. Generate synthetic candles so you can try the backtester offline
PYTHONPATH=. python scripts/generate_sample_data.py --epic EURUSD --bars 4000
PYTHONPATH=. python scripts/generate_sample_data.py --epic GBPUSD --bars 4000 --seed 7 --start-price 1.27

# 3. Backtest and write CSV + HTML reports into ./reports
PYTHONPATH=. python -m forex_bot.cli backtest --report-dir reports

# 4. Run the test suite (stdlib unittest — no pytest required)
python -m unittest discover -s tests
```

## Connecting to Capital.com

1. Create a **demo** account and generate an API key under
   **Settings → API integrations**.
2. Copy `.env.example` to `.env` and fill in `CAPITAL_IDENTIFIER`,
   `CAPITAL_PASSWORD`, and `CAPITAL_API_KEY`. Keep `CAPITAL_ENVIRONMENT=demo`.
3. Install the connectivity extras:

   ```bash
   pip install -e .            # core + requests + websocket-client + pyyaml
   pip install -e ".[research]"  # optional: pandas/numpy/pyarrow/matplotlib
   ```

4. Download real history, then backtest:

   ```bash
   forex-bot download
   forex-bot backtest --report-dir reports
   ```

5. Run live against **demo** (paper) — recommended for several weeks before any
   real capital:

   ```bash
   forex-bot demo
   ```

6. Only after thorough demo validation, with reduced sizing and conservative
   limits in `config/config.yaml`:

   ```bash
   forex-bot live    # LIVE — real funds at risk
   ```

## Configuration

- **Secrets** live in `.env` (never committed). See `.env.example`.
- **Trading config** lives in `config/config.yaml` (see `config.example.yaml`):
  instruments/timeframes, the selected strategy and its params, simulated costs,
  and the risk framework (`risk_per_trade`, `max_position_pct`,
  `max_open_positions`, `max_daily_loss_pct`).

## Writing a strategy

Subclass `StrategyBase`, implement `on_candle`, and register it:

```python
from forex_bot.models import Signal, SignalType
from forex_bot.strategy.base import StrategyBase

class MyStrategy(StrategyBase):
    warmup = 50
    def on_candle(self, candle, context):
        if len(context.closes) < self.warmup:
            return None
        # ... compute indicators from context.closes/highs/lows ...
        return Signal(candle.epic, SignalType.ENTER_LONG, candle.timestamp)
```

Add it to `STRATEGY_REGISTRY` in `forex_bot/strategy/__init__.py`, then select it
by name in `config.yaml`.

## Safety notes

- The risk manager **halts new entries** once the configured daily loss limit is
  breached, and caps both per-trade risk and per-position notional.
- Backtests apply spread, slippage and commission so results are not optimistic
  fills. Validate every strategy on demo before going live.
- This is a foundation, not a finished trading system. Phases 4–5 (extended demo
  soak, alerting, dashboards, a kill switch, broker-state reconciliation under
  failure) should be hardened before risking real capital.

## License

MIT.
