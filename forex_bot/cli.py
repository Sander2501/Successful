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


def _make_client(creds, args):
    """Construct a REST client with a shared session-token cache, so repeated
    CLI invocations reuse one session instead of re-logging-in (avoids 429)."""
    from .api.rest_client import CapitalRestClient
    cache = Path(args.data_dir).parent / "db" / f"session_{creds.environment}.json"
    return CapitalRestClient(creds, session_cache_path=str(cache))


def _instruments_from_args(config: TradingConfig, args):
    """Instruments to operate on: a comma-separated --epics override, else the
    config's instruments. Lets you explore pairs without editing config."""
    from .config import InstrumentConfig
    epics = getattr(args, "epics", None)
    if epics:
        tf = getattr(args, "timeframe", None) or (
            config.instruments[0].timeframe if config.instruments else "MINUTE_15")
        return [InstrumentConfig(epic=e.strip(), timeframe=tf)
                for e in epics.split(",") if e.strip()]
    return config.instruments


# --------------------------------------------------------------------------- #
def cmd_download(args: argparse.Namespace) -> int:
    from .data.storage import CandleStore

    config = _load_config(args.config)
    creds = CapitalCredentials.from_env()
    client = _make_client(creds, args)
    store = CandleStore(args.data_dir)

    for inst in _instruments_from_args(config, args):
        log.info("downloading", extra={"epic": inst.epic, "tf": inst.timeframe,
                                       "max_bars": args.max_bars})
        # Page backward past the ~1000/request cap when more is requested.
        candles = client.get_historical_prices_paged(
            inst.epic, inst.timeframe, total=args.max_bars
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
    from .research.walkforward import DEFAULT_GRIDS
    from .strategy import STRATEGY_REGISTRY

    config = _load_config(args.config)
    opt = config.optimize or {}

    store = CandleStore(args.data_dir)
    candles_by_epic = {}
    for inst in _instruments_from_args(config, args):
        bars = store.load(inst.epic, inst.timeframe)
        if not bars:
            log.error("no stored candles; run 'download' first",
                      extra={"epic": inst.epic, "tf": inst.timeframe})
            return 2
        candles_by_epic[inst.epic] = bars

    if args.all_strategies:
        strategies = list(STRATEGY_REGISTRY)
    elif args.strategy:
        strategies = [args.strategy]
    else:
        strategies = [config.strategy]

    grids = opt.get("param_grids", {}) or {}

    def grid_for(name: str) -> dict:
        return grids.get(name) or DEFAULT_GRIDS.get(name) or opt.get("param_grid", {})

    results = []
    for name in strategies:
        grid = grid_for(name)
        if not grid:
            log.warning("no parameter grid for strategy; skipping", extra={"strategy": name})
            continue
        try:
            res = walk_forward(
                candles_by_epic, config, name, grid,
                is_bars=int(opt.get("is_bars", 1500)),
                oos_bars=int(opt.get("oos_bars", 500)),
                step_bars=opt.get("step_bars"),
                metric=opt.get("metric", "sharpe"),
                warmup_bars=int(opt.get("warmup_bars", 250)),
                min_trades=int(opt.get("min_trades", 5)),
            )
        except ValueError as exc:
            log.error("walk-forward failed", extra={"strategy": name, "error": str(exc)})
            continue
        results.append(res)
        if not args.all_strategies:
            print("\n" + res.to_text() + "\n")

    if args.all_strategies:
        print("\n" + _strategy_comparison(results) + "\n")
    return 0


def _strategy_comparison(results) -> str:
    """Rank strategies by out-of-sample result and give a go/no-go verdict."""
    if not results:
        return "No strategies could be evaluated (insufficient data or no grids)."
    ranked = sorted(results, key=lambda r: r.combined_oos_return_pct, reverse=True)
    lines = [
        "Walk-forward comparison (out-of-sample, real data):",
        f"  {'strategy':<20} {'OOS ret%':>9} {'+folds%':>8} {'OOS PF':>7} {'trades':>7}",
    ]
    for r in ranked:
        lines.append(
            f"  {r.strategy:<20} {r.combined_oos_return_pct:>9.2f} "
            f"{r.pct_positive_folds:>8.0f} {r.combined_profit_factor:>7.2f} "
            f"{r.total_oos_trades:>7}"
        )
    # A defensible "edge" must be OOS-positive, win the majority of folds, and
    # have a profit factor above 1 with a non-trivial sample.
    survivors = [
        r for r in ranked
        if r.combined_oos_return_pct > 0
        and r.pct_positive_folds >= 50
        and r.combined_profit_factor > 1.0
        and r.total_oos_trades >= 20
    ]
    lines.append("")
    if survivors:
        names = ", ".join(r.strategy for r in survivors)
        lines.append(f"VERDICT: candidate edge survived out-of-sample -> {names}")
        lines.append("Validate further (more history, more pairs) before risking capital.")
    else:
        lines.append("VERDICT: no strategy showed a robust out-of-sample edge on this data.")
        lines.append("Do NOT trade live. Try a higher timeframe (HOUR_4/DAY), more history, "
                     "or different instruments — or accept there is no tradeable edge here.")
    return "\n".join(lines)


def _format_search(data: dict) -> str:
    markets = (data or {}).get("markets", []) if isinstance(data, dict) else []
    if not markets:
        return "No markets found."
    lines = [f"  {'epic':<22} {'name':<30} {'type':<10} status"]
    for m in markets[:50]:
        lines.append(
            f"  {str(m.get('epic', '')):<22} "
            f"{str(m.get('instrumentName') or '')[:30]:<30} "
            f"{str(m.get('instrumentType', '')):<10} {m.get('marketStatus', '')}"
        )
    return "\n".join(lines)


def cmd_search(args: argparse.Namespace) -> int:
    creds = CapitalCredentials.from_env()
    client = _make_client(creds, args)
    client.ensure_session()  # reuses a cached session token when fresh
    data = client.search_markets(args.term)
    print(f"\nMarkets matching '{args.term}':")
    print(_format_search(data) + "\n")
    print("Add the epic(s) you want to config.yaml under `instruments`, then "
          "`download` and `screen`/`optimize`.")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    from .data.storage import CandleStore
    from .research import screen_pairs, screen_report

    config = _load_config(args.config)
    store = CandleStore(args.data_dir)
    candles_by_epic = {}
    for inst in _instruments_from_args(config, args):
        bars = store.load(inst.epic, inst.timeframe)
        if bars:
            candles_by_epic[inst.epic] = bars
        else:
            log.warning("no stored candles; run 'download' first",
                        extra={"epic": inst.epic, "tf": inst.timeframe})
    if len(candles_by_epic) < 2:
        log.error("need at least two downloaded instruments to screen pairs")
        return 2
    print("\n" + screen_report(screen_pairs(candles_by_epic)) + "\n")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    from .preflight import all_passed, report_text, run_preflight

    config = _load_config(args.config)
    creds = CapitalCredentials.from_env()
    if args.environment:
        creds.environment = args.environment

    log.info("running preflight", extra={"environment": creds.environment,
                                         "test_order": bool(args.test_order)})
    results = run_preflight(config, creds, client=_make_client(creds, args),
                            test_order=args.test_order)
    print("\n" + report_text(results) + "\n")
    return 0 if all_passed(results) else 1


def cmd_run(args: argparse.Namespace, environment: str) -> int:
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

    client = _make_client(creds, args)
    strategy = build_strategy(config.strategy, config.strategy_params)
    from .strategy.portfolio_base import PortfolioStrategy
    if isinstance(strategy, PortfolioStrategy):
        log.error("portfolio strategies (e.g. spread_reversion) are not yet supported "
                  "in live mode; validate them with backtest/optimize first")
        return 2
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

    def _add_epics(parser):
        parser.add_argument("--epics", help="comma-separated epics to use instead of "
                            "config instruments, e.g. EURGBP,EURCHF,AUDNZD")
        parser.add_argument("--timeframe", help="timeframe for --epics (default: config)")

    d = sub.add_parser("download", help="download historical candles")
    d.add_argument("--max-bars", type=int, default=1000)
    _add_epics(d)
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("backtest", help="run a backtest on stored candles")
    b.add_argument("--report-dir", default="reports", help="where to write reports")
    b.set_defaults(func=cmd_backtest)

    o = sub.add_parser("optimize", help="walk-forward optimize the configured strategy")
    o.add_argument("--strategy", help="override the strategy to optimize")
    o.add_argument("--all-strategies", action="store_true",
                   help="walk-forward every registered strategy and rank them")
    _add_epics(o)
    o.set_defaults(func=cmd_optimize)

    se = sub.add_parser("search", help="search Capital.com for market epics by term")
    se.add_argument("term", help="search term, e.g. EURGBP or 'Australian Dollar'")
    se.set_defaults(func=cmd_search)

    sc = sub.add_parser("screen", help="rank instrument pairs by mean-reversion (spread) quality")
    _add_epics(sc)
    sc.set_defaults(func=cmd_screen)

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
