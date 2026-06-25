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
    print(report.to_text())
    print()

    if args.report_dir:
        out = Path(args.report_dir)
        export_trades_csv(result.trades, out / "trades.csv")
        export_equity_csv(result.equity_curve, out / "equity.csv")
        export_html_report(result, report, out / "report.html")
        log.info("reports written", extra={"dir": str(out)})
    return 0


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

    client = CapitalRestClient(creds)
    strategy = build_strategy(config.strategy, config.strategy_params)
    engine = LiveTradingEngine(strategy, config, client)
    try:
        engine.start()
    except KeyboardInterrupt:
        log.info("shutting down")
        engine.stop()
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
