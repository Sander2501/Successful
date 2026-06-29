# Successful - Forex Trading Bot for Capital.com

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
├── indicators.py        Pure indicators: SMA, EMA, RSI, ATR, ADX/DMI, Donchian
├── strategy/            Pluggable strategies (registry-based)
│   ├── base.py          StrategyBase + StrategyContext
│   ├── ema_crossover.py EMA crossover trend-follower (trend + ADX filters)
│   ├── donchian_breakout.py Turtle-style channel breakout (ADX-gated)
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
├── research/            Edge discovery & validation
│   └── walkforward.py   Walk-forward optimization + out-of-sample evaluation
├── state/               Durable, restart-safe state
│   └── store.py         SQLite store: positions + risk high-water mark/kill flag
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

4. **Preflight** — validate the whole broker path before trading. This is the
   single command to run from your deployment environment with real demo
   credentials; everything except `--test-order` is read-only:

   ```bash
   forex-bot preflight                 # login, equity, market data, history, positions
   forex-bot preflight --test-order    # also opens + closes one minimal demo position
   ```

   It prints a pass/fail line per step and exits non-zero on any failure, so it
   doubles as a deploy gate:

   ```
   Preflight: 8/8 checks passed
     [PASS] login — environment=demo
     [PASS] account/equity — equity=9987.65
     [PASS] market details — EURUSD status=TRADEABLE
     [PASS] history — EURUSD MINUTE_15: 10 bars, last close=1.10050
     ...
   ```

5. Download real history, then backtest:

   ```bash
   forex-bot download
   forex-bot backtest --report-dir reports
   ```

6. Run live against **demo** (paper) — recommended for several weeks before any
   real capital:

   ```bash
   forex-bot demo
   ```

7. Only after thorough demo validation, with reduced sizing and conservative
   limits in `config/config.yaml`:

   ```bash
   forex-bot preflight --environment live   # read-only sanity check on live
   forex-bot live                           # LIVE — real funds at risk
   ```

> **Why a separate preflight?** The live engine's broker interaction (auth,
> position reconciliation, account-equity parsing, order placement) can't be
> exercised by the offline test suite. Preflight validates it against the real
> API. The orchestration *logic* around it (warmup → candle → signal → risk →
> order → fill → persist, plus the kill-switch flatten and restart paths) is
> covered by an integration test that drives the engine with a fake broker, so
> only the genuine network/auth surface needs a live run.

## Configuration

- **Secrets** live in `.env` (never committed). See `.env.example`.
- **Trading config** lives in `config/config.yaml` (see `config.example.yaml`):
  instruments/timeframes, the selected strategy and its params, simulated costs,
  and the risk framework (`risk_per_trade`, `max_position_pct`,
  `max_open_positions`, `max_daily_loss_pct`).

### Correlation-aware risk

EUR/USD, GBP/USD and AUD/USD are all *short USD* when long — several "different"
pairs can secretly be one big USD bet. The risk manager decomposes every FX
position into per-currency notionals and enforces:

- `max_currency_exposure_pct` — cap net exposure to any single currency as a
  fraction of equity. Because one position's notional is already
  `max_position_pct × equity`, this value must be **≥ `max_position_pct`** or it
  blocks the first trade. On the bundled two-pair backtest, tightening the USD
  cap from 100% → 75% cut max drawdown 3.18% → 2.39% by refusing to stack
  concurrent USD bets; on genuinely correlated live pairs the benefit is larger.
- `max_positions_per_currency` — optional cap on how many open positions may
  share a currency.

Currencies are auto-parsed from 6-letter epics (`EURUSD` → EUR/USD); set
`base_currency`/`quote_currency` on an instrument for non-standard epics.

### Data-driven correlation groups

Shared currency codes miss correlations that don't share a ticker (EUR/USD vs the
dollar index) and sign (EUR/USD vs USD/CHF move *opposite*). With
`correlation_threshold` set, the engine estimates pairwise return correlation
from the candles, clusters instruments above the threshold, and caps **net
directional** exposure per cluster (`max_correlated_exposure_pct`,
`max_positions_per_group`) — sign-aware, so a long EUR/USD, long GBP/USD and
*short* USD/CHF are correctly counted as one big bet. Correlations are computed
once per backtest and from warmup history when live.

### Portfolio kill switch

`max_total_drawdown_pct` is the hard backstop beyond the daily-loss halt: once
equity falls that far below its all-time high-water mark, the engine **flattens
every position and stops opening new ones** for the rest of the run. Live, equity
is refreshed from the broker account so the switch reacts to real balance.
`RiskManager.kill()` exposes the same as a manual operator action.

### Volatility-targeted sizing

With `sizing_mode: vol_target`, positions are sized so a 1-ATR move equals
`vol_target_pct` of equity — so a quiet pair and a wild one contribute **equal
risk** instead of equal notional. Verified: at ATR 0.02 / 0.05 / 0.10 the sizer
returns 5000 / 2000 / 1000 units, each a 100 (1% of equity) move per ATR. Falls
back to fixed-fractional stop-distance sizing when no ATR is available.

## Durable state (restart safety)

Set `state_db` to a SQLite path and live trading becomes restartable without
losing what it knows (a plan requirement). Persisted across restarts:

- **open positions** with their protective levels and broker deal ids, and
- **risk state** — the equity high-water mark, the kill-switch flag, and the
  daily-loss bookkeeping.

This matters because the kill switch tracks drawdown from a high-water mark; a
crash-and-restart without persistence resets that mark, so a bot that had already
tripped (or nearly tripped) its kill switch would happily resume. On startup the
engine restores risk state, reconciles live positions against the broker (the
source of truth) while recovering stop/target metadata from the store, and halts
immediately if the restored state was killed. State is written on every
open/close and equity refresh.

Two supporting hardening changes:

- **Periodic correlation refresh** (`correlation_refresh_bars`) — correlations
  drift and break in a crisis, so live trading re-estimates them every N candles
  instead of trusting the warmup window forever.
- **Robust equity parsing** — the broker account balance is read defensively
  (preferred account first, then several known balance fields) so a minor schema
  variation can't silently disable the drawdown limits.

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

### Reducing whipsaw (EMA crossover)

A bare EMA crossover over-trades in ranging markets — every wiggle round-trips
the spread. Three **stateless** filters (config under `strategy_params`) cut this:

- `trend_filter` — a long-period regime EMA; only take longs above it / shorts
  below it. On the bundled sample this drops trades 286 → 138, lifts win rate
  33% → 39% and profit factor 0.81 → 0.95.
- `adx_period` / `adx_threshold` — an **ADX regime gate**: only enter when trend
  *strength* exceeds the threshold. This is the main edge lever for a trend
  system — it keeps it out of range-bound chop where crossovers bleed.
- `min_separation_pct` — discard crosses where the EMAs are barely apart. This
  is a **fraction of price**, so it is volatility-sensitive: too large a value
  silences all trades. Tune it per market (`0.0` disables it).

### Donchian breakout (`donchian_breakout`)

A classic Turtle-style channel breakout: go long on a break above the highest
high of the last `entry` bars, short below the lowest low, exit on the opposite
`exit` channel, with ATR stops and the same ADX regime gate. Time-series
momentum/breakout is one of the more robustly documented cross-asset anomalies,
which makes it a sensible second strategy to validate rather than a curve fit.

### Spread / pairs mean-reversion (`spread_reversion`)

A **relative-value** strategy and a genuinely different return source from the
directional ones above. For two cointegrated instruments it forms the spread
`s = ln(Pₐ) − β·ln(P_b)` (β a rolling hedge ratio), and when the spread's
z-score is extreme it bets on reversion — short the rich leg, long the cheap leg,
exit as the z-score returns to zero (with a `stop_z` for regime breaks). It is
roughly market-neutral, so a shared-USD move that lifts both legs largely
cancels. This is a `PortfolioStrategy` (it sees both instruments at once); the
backtester runs it on a bar-synchronized loop. Directional TA on liquid FX
majors is close to efficient — relative value is where a small, defensible edge
is more plausible — but it is **not** assumed: run `optimize --all-strategies`
and let the out-of-sample verdict decide. Currently backtest/research only (not
yet wired into live execution).

## Finding a real edge (walk-forward validation)

The honest question is not "did it make money on this data?" but "does the edge
survive on data the optimizer never saw?" `forex-bot optimize` answers that with
**walk-forward analysis**: it rolls through the history, tunes parameters on each
in-sample window, and scores them **only** on the following out-of-sample window.

```bash
forex-bot optimize                    # the configured strategy
forex-bot optimize --strategy donchian_breakout
forex-bot optimize --all-strategies   # sweep every strategy and rank them
```

It reports per-fold chosen parameters plus pooled out-of-sample return, percent
of positive folds, and profit factor. `--all-strategies` ranks them and prints a
**go/no-go verdict** — it only calls something a "candidate edge" if it is
OOS-positive, wins the majority of folds, and has profit factor > 1 on a
non-trivial sample. If nothing qualifies it says so plainly: do not trade live.

> **Timeframe matters.** Low/medium-frequency trend and breakout systems are
> designed for HOUR_4 / DAY bars, not 15-minute noise where spread dominates. If
> a sweep finds no edge on `MINUTE_15`, switch the instruments' `timeframe` to
> `HOUR_4`, `download` again, and re-run — that is the single most likely place
> an edge appears (or to confirm there genuinely isn't one).

### Choosing pairs for spread trading (avoid data-snooping)

Testing many pairs and keeping the winner is data-snooping — with enough pairs
something looks good by chance. Pick candidates by a *statistical* criterion
first, then let walk-forward judge the shortlist:

```bash
forex-bot screen        # ranks configured instruments by spread mean-reversion
```

For each pair it fits a hedge ratio, builds the spread, and reports the
**Engle-Granger ADF** statistic (is the spread stationary?), the **half-life**
(how fast it reverts), and the return correlation. A pair is only flagged
`TEST IT` if it is cointegrated **and** co-moving (a correlation floor rejects
spurious cointegration with an unrelated series) **and** reverts at a tradeable
speed. EUR/USD and GBP/USD are weak candidates because they are both USD legs;
economically-linked **crosses** (EUR/GBP, EUR/CHF, AUD/NZD) are more promising.
Screening is necessary, not sufficient — the spread still has to clear costs
out-of-sample, which is what `optimize` decides.

### Honest costs and the one-shot holdout

Three tools keep the search honest once something looks promising:

- **Per-instrument costs.** Each instrument carries its own `spread_points` and
  `value_per_point`, threaded through execution, PnL and risk. A basket mixing
  EUR/USD (pip 0.0001) and USD/JPY (pip 0.01) is now priced correctly instead of
  charging every leg the first instrument's spread.
- **Measure real spreads.** `forex-bot spreads` pulls the live bid/ask per
  instrument so cost assumptions are real numbers, not guesses — paste them into
  each instrument's `spread_points`.
- **`forex-bot optimize --cost-stress`** re-runs at 1×–5× spread; a spread-thin
  "edge" visibly collapses as costs rise.
- **`forex-bot holdout`** optimizes on the early data and tests **once** on a
  held-out final slice the optimizer never saw — the only real cure for the
  data-snooping that creeps in after many `optimize` runs. Run it once; re-tuning
  and re-running turns the holdout into just more snooping.

A candidate worth real money should: pass `--all-strategies` with a real sample,
survive `--cost-stress` at your measured spreads, and clear a single `holdout` —
in that order. Sanity check on synthetic data — the tool
correctly tells edge from noise:

| Data | Combined OOS return | Positive folds | OOS profit factor |
|------|--------------------:|---------------:|------------------:|
| Random walk (no edge) | −0.68% | 25% | 0.73 |
| Trending (real edge)  | +2.59% | 100% | 4.20 |

Generate those two datasets yourself:

```bash
python scripts/generate_sample_data.py --epic NOISE --bars 3000 --mode random
python scripts/generate_sample_data.py --epic TREND --bars 3000 --mode trending
```

> The `trending` mode injects a synthetic, genuinely-exploitable trend purely to
> demonstrate that the harness detects an edge when one exists. It is **not** a
> claim about real markets. An edge that only appears in-sample, or evaporates
> out-of-sample, was never real — that is exactly what this tool is for.

### Canonical rejection report and the research registry

`forex-bot report` runs **every** strategy through one identical walk-forward
config and writes a single comparable table — OOS return, PF, positive folds,
trades, avg R, worst month, worst fold, per-instrument contribution, and a
**deterministic PASS/FAIL verdict** — to `results/strategy_report.{csv,md}`
(gitignored). The verdict rules live in `forex_bot/research/verdicts.py` and are
config-driven via a `report:` block with strict defaults, so rejection is
mechanical, not vibes.

```bash
forex-bot report                       # screen every strategy, write results/
forex-bot report --skip-frozen --record  # skip frozen ideas; record outcomes
```

The **research registry** (`research/registry.json`, committed) freezes failed
ideas so they are not accidentally re-opened. State is tracked **per market
setup** — a `(timeframe, instrument-universe)` key — so a `holdout-fail` on
MINUTE_15 crosses does *not* freeze the same strategy on HOUR_4:

```bash
forex-bot registry                     # show the ledger
forex-bot holdout --strategy rsi_reversion --record   # records the outcome
forex-bot registry --strategy rsi_reversion --status holdout-fail \
  --epics EURUSD,GBPUSD --timeframe HOUR_4 --note "clean test failed"
```

`report` warns when a strategy is frozen for the current setup (and drops it with
`--skip-frozen`), turning the pipeline from a backtest playground into a research
system with memory. Statuses: `candidate`, `screen-fail`, `holdout-fail`,
`holdout-pass`, `forward-test`, `live`, `retired` — `holdout-fail` and `retired`
freeze the setup.

A holdout never auto-grants `forward-test`. `holdout --record` only ever writes
`holdout-fail` (the edge died), `candidate` (positive but thin/inconclusive — too
few trades or PF below the screen bar), or `holdout-pass` (an adequate-sample
clean pass). `forward-test` is a deliberate decision you make **after**
cost-stress — a barely-positive holdout (e.g. +0.04%, PF 1.06, 14 trades) is
noise, not a green light.

## Performance metrics

Return-based statistics (Sharpe, Sortino, annual volatility) are computed on the
equity curve **resampled to one point per calendar day**, annualized with a
252-trading-day year. The raw curve is sampled once per candle, so annualizing
intra-day returns by their native frequency wildly inflates volatility — daily
resampling keeps the figures comparable to how strategies are normally quoted.
The report also breaks out `trading_days`, `trades_per_day`, and `total_fees`.

## Safety notes

- The risk manager **halts new entries** once the configured daily loss limit is
  breached, and caps both per-trade risk and per-position notional.
- Backtests apply spread, slippage and commission so results are not optimistic
  fills. Validate every strategy on demo before going live.
- This is a foundation, not a finished trading system. Phases 4–5 (extended demo
  soak, alerting, dashboards, a kill switch, broker-state reconciliation under
  failure) should be hardened before risking real capital.

## Security / Dependency scanning

**You're missing a lockfile.**  
To give Aikido a complete view of your dependencies and related vulnerabilities, please add and commit a lockfile (for example, `poetry.lock`, `Pipfile.lock`, or `requirements.txt` with pinned versions).

## License

MIT.
