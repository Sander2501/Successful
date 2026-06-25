"""Command-line entry point.

Subcommands:
  download   Fetch historical candles from Capital.com into the local store.
  backtest   Replay stored candles through a strategy and print/export a report.
  demo|live  Run the live trading engine against the demo or live environment.

The ``backtest`` command runs purely on the stdlib core; ``download`` and the
live commands require the broker dependencies (requests / websocket-client) and
valid credentials in ``.env``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CapitalCredentials, TradingConfig
from .logging_setup import get_logger, setup_logging

log = get_logger("forex_bot.cli")


def _load_config(path: str) -> TradingConfig:
    if not Path(path).exists():
        log.error("config file not found", extra={"path": path})
        sys.exit(2)
    return TradingConfig.from_yaml(path)


# --------------------------------------------------------------------------- #
def cmd_download(args: argparse.Namespace) -> int:
    from .api.rest_client import CapitalRestClient
    from .data.storage import CandleStore

    config = _load_config(args.config)
    creds = CapitalCredentials.from_env()
    client = CapitalRestClient(creds)
    store = CandleStore(args.data_dir)

    for inst in config.instruments:
        log.info("downloading", extra={"epic": inst.epic, "tf": inst.timeframe})
        candles = client.get_historical_prices(
            inst.epic, inst.timeframe, max_bars=args.max_bars
        )
        report = store.quality_report(candles, inst.timeframe)
        log.info("quality", extra=report)
        store.save(inst.epic, inst.timeframe, candles)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest.engine import Backtester
    from .backtest.metrics import compute_metrics
    from .backtest.reporting import (
        export_equity_csv,
        export_html_report,
        export_trades_csv,
    )
    from .data.storage import CandleStore
    from .strategy import build_strategy

    config = _load_config(args.config)
    store = CandleStore(args.data_dir)

    candles_by_epic = {}
    for inst in config.instruments:
        bars = store.load(inst.epic, inst.timeframe)
        if not bars:
            log.error("no stored candles; run 'download' first",
                      extra={"epic": inst.epic, "tf": inst.timeframe})
            return 2
        candles_by_epic[inst.epic] = bars

    strategy = build_strategy(config.strategy, config.strategy_params)
    result = Backtester(strategy, config).run(candles_by_epic)
    report = compute_metrics(result.equity_curve, result.trades)

    print(f"\n=== Backtest: {strategy.name} ===")
    print(_active_controls_summary(config))
    print()
    print(report.to_text())
    print()

    if args.report_dir:
        out = Path(args.report_dir)
        export_trades_csv(result.trades, out / "trades.csv")
        export_equity_csv(result.equity_curve, out / "equity.csv")
        export_html_report(result, report, out / "report.html")
        log.info("reports written", extra={"dir": str(out)})
    return 0


def _active_controls_summary(config) -> str:
    """Human-readable summary of the strategy params and risk controls actually
    in effect, with warnings when key safeguards are disabled. Makes a stale or
    minimal config impossible to miss in the output."""
    p = config.strategy_params or {}
    r = config.risk
    lines = [
        f"Strategy params : {p if p else '(defaults — no filters configured)'}",
        f"Sizing          : {r.sizing_mode}"
        + (f" (vol_target_pct={r.vol_target_pct})" if r.sizing_mode == "vol_target" else ""),
        f"Per-position cap: {r.max_position_pct:.0%} equity   "
        f"daily-loss halt: {r.max_daily_loss_pct:.0%}",
        f"Currency cap    : {r.max_currency_exposure_pct:.0%}   "
        f"correlation: "
        + (f"group>={r.correlation_threshold} cap {r.max_correlated_exposure_pct:.0%}"
           if r.correlation_threshold is not None else "OFF"),
        f"Kill switch     : "
        + (f"{r.max_total_drawdown_pct:.0%} total drawdown"
           if r.max_total_drawdown_pct < 1.0 else "OFF"),
    ]
    warnings = []
    if not p:
        warnings.append("strategy is running on DEFAULTS — trend/ADX filters are off")
    if r.correlation_threshold is None:
        warnings.append("correlation grouping is OFF")
    if r.max_total_drawdown_pct >= 1.0:
        warnings.append("portfolio kill switch is OFF")
    if warnings:
        lines.append("!! " + "; ".join(warnings))
        lines.append("!! if unexpected, refresh config.yaml from config.example.yaml")
    return "\n".join(lines)


def cmd_optimize(args: argparse.Namespace) -> int:
    from .data.storage import CandleStore
    from .research import walk_forward

    config = _load_config(args.config)
    opt = config.optimize or {}
    grid = opt.get("param_grid", {})
    if not grid:
        log.error("config has no 'optimize.param_grid'; nothing to search")
        return 2

    store = CandleStore(args.data_dir)
    candles_by_epic = {}
    for inst in config.instruments:
        bars = store.load(inst.epic, inst.timeframe)
        if not bars:
            log.error("no stored candles; run 'download' first",
                      extra={"epic": inst.epic, "tf": inst.timeframe})
            return 2
        candles_by_epic[inst.epic] = bars

    result = walk_forward(
        candles_by_epic,
        config,
        config.strategy,
        grid,
        is_bars=int(opt.get("is_bars", 1500)),
        oos_bars=int(opt.get("oos_bars", 500)),
        step_bars=opt.get("step_bars"),
        metric=opt.get("metric", "sharpe"),
        warmup_bars=int(opt.get("warmup_bars", 250)),
        min_trades=int(opt.get("min_trades", 5)),
    )
    print("\n" + result.to_text() + "\n")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    from .preflight import all_passed, report_text, run_preflight

    config = _load_config(args.config)
    creds = CapitalCredentials.from_env()
    if args.environment:
        creds.environment = args.environment

    log.info("running preflight", extra={"environment": creds.environment,
                                         "test_order": bool(args.test_order)})
    results = run_preflight(config, creds, test_order=args.test_order)
    print("\n" + report_text(results) + "\n")
    return 0 if all_passed(results) else 1


def cmd_run(args: argparse.Namespace, environment: str) -> int:
    from .api.rest_client import CapitalRestClient
    from .live_engine import LiveTradingEngine
    from .strategy import build_strategy

    config = _load_config(args.config)
    creds = CapitalCredentials.from_env()
    if creds.environment != environment:
        log.warning("overriding environment from config/env",
                    extra={"requested": environment, "env_value": creds.environment})
        creds.environment = environment

    if environment == "live":
        log.warning("LIVE TRADING ENABLED — real funds at risk")

    state_store = None
    if config.state_db:
        from .state.store import StateStore
        state_store = StateStore(config.state_db)
        log.info("state persistence enabled", extra={"db": config.state_db})

    client = CapitalRestClient(creds)
    strategy = build_strategy(config.strategy, config.strategy_params)
    engine = LiveTradingEngine(strategy, config, client, state_store=state_store)
    try:
        engine.start()
    except KeyboardInterrupt:
        log.info("shutting down")
        engine.stop()
    finally:
        if state_store is not None:
            state_store.close()
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forex-bot", description=__doc__)
    p.add_argument("--config", default="config/config.yaml", help="path to YAML config")
    p.add_argument("--data-dir", default="data/historical", help="historical data dir")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download historical candles")
    d.add_argument("--max-bars", type=int, default=1000)
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("backtest", help="run a backtest on stored candles")
    b.add_argument("--report-dir", default="reports", help="where to write reports")
    b.set_defaults(func=cmd_backtest)

    o = sub.add_parser("optimize", help="walk-forward optimize the configured strategy")
    o.set_defaults(func=cmd_optimize)

    pf = sub.add_parser("preflight", help="validate the live broker path with real credentials")
    pf.add_argument("--environment", choices=["demo", "live"], default="demo")
    pf.add_argument("--test-order", action="store_true",
                    help="also place and immediately close one minimal demo position")
    pf.set_defaults(func=cmd_preflight)

    demo = sub.add_parser("demo", help="run live engine against the demo environment")
    demo.set_defaults(func=lambda a: cmd_run(a, "demo"))

    live = sub.add_parser("live", help="run live engine against the LIVE environment")
    live.set_defaults(func=lambda a: cmd_run(a, "live"))

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
