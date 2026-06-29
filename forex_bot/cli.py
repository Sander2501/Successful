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
        "Kill switch     : "
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

    def run_wf(name, grid, cost_multiplier=1.0):
        return walk_forward(
            candles_by_epic, config, name, grid,
            is_bars=int(opt.get("is_bars", 1500)),
            oos_bars=int(opt.get("oos_bars", 500)),
            step_bars=opt.get("step_bars"),
            metric=opt.get("metric", "sharpe"),
            warmup_bars=int(opt.get("warmup_bars", 250)),
            min_trades=int(opt.get("min_trades", 5)),
            cost_multiplier=cost_multiplier,
            select_top_n=args.select_top,
            select_metric=args.select_metric,
        )

    if args.cost_stress:
        print("\n" + _cost_stress_report(strategies, grid_for, run_wf, config) + "\n")
        return 0

    results = []
    for name in strategies:
        grid = grid_for(name)
        if not grid:
            log.warning("no parameter grid for strategy; skipping", extra={"strategy": name})
            continue
        try:
            res = run_wf(name, grid)
        except ValueError as exc:
            log.error("walk-forward failed", extra={"strategy": name, "error": str(exc)})
            continue
        results.append(res)
        if not args.all_strategies:
            print("\n" + res.to_text() + "\n")

    if args.all_strategies:
        print("\n" + _strategy_comparison(results, config) + "\n")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from pathlib import Path as _Path

    from .data.storage import CandleStore
    from .research import (
        ResearchRegistry,
        run_strategy_report,
        setup_key,
        thresholds_from_config,
        to_csv,
        to_markdown,
    )
    from .research.walkforward import DEFAULT_GRIDS
    from .strategy import STRATEGY_REGISTRY

    config = _load_config(args.config)
    opt = config.optimize or {}

    instruments = _instruments_from_args(config, args)
    store = CandleStore(args.data_dir)
    candles_by_epic = {}
    for inst in instruments:
        bars = store.load(inst.epic, inst.timeframe)
        if not bars:
            log.error("no stored candles; run 'download' first",
                      extra={"epic": inst.epic, "tf": inst.timeframe})
            return 2
        candles_by_epic[inst.epic] = bars

    setup = setup_key(instruments)
    registry = ResearchRegistry.load(args.registry)
    strategies = [args.strategy] if args.strategy else list(STRATEGY_REGISTRY)

    # Discipline layer: warn (or skip) strategies already frozen for THIS setup so
    # a failed idea is not silently re-opened by re-running with new parameters.
    kept = []
    for name in strategies:
        if registry.is_frozen(name, setup):
            entry = registry.get(name, setup)
            log.warning("strategy is frozen for this setup",
                        extra={"strategy": name, "setup": setup,
                               "status": entry.status, "note": entry.note})
            if args.skip_frozen:
                continue
        kept.append(name)
    strategies = kept

    grids = opt.get("param_grids", {}) or {}

    def grid_for(name: str) -> dict:
        return grids.get(name) or DEFAULT_GRIDS.get(name) or opt.get("param_grid", {})

    # Walk-forward only (no holdout / cost-stress): one identical config per strategy.
    wf_kwargs = dict(
        is_bars=int(opt.get("is_bars", 1500)),
        oos_bars=int(opt.get("oos_bars", 500)),
        step_bars=opt.get("step_bars"),
        metric=opt.get("metric", "sharpe"),
        warmup_bars=int(opt.get("warmup_bars", 250)),
        min_trades=int(opt.get("min_trades", 5)),
        select_top_n=args.select_top,
        select_metric=args.select_metric,
    )
    thresholds = thresholds_from_config(config)

    def _on_skip(name: str, reason: str) -> None:
        log.warning("skipping strategy", extra={"strategy": name, "reason": reason})

    report = run_strategy_report(
        candles_by_epic, config, strategies, grid_for, wf_kwargs, thresholds,
        on_skip=_on_skip,
    )
    if not report.rows:
        log.error("no strategies could be evaluated (no grids or insufficient data)")
        return 2

    out_dir = _Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "strategy_report.csv"
    md_path = out_dir / "strategy_report.md"
    csv_path.write_text(to_csv(report))
    md_text = to_markdown(report)
    md_path.write_text(md_text)

    print("\n" + md_text + "\n")
    print(f"wrote {csv_path} and {md_path}")

    if args.record:
        # Record screen outcomes, but never overwrite a FROZEN (holdout-fail/
        # retired) entry — a screen pass must not silently re-open a spent idea.
        for r in report.rows:
            status = "candidate" if r.passed else "screen-fail"
            registry.set_status(r.strategy, setup, status,
                                note="auto: walk-forward screen",
                                allow_overwrite_frozen=False)
        registry.save(args.registry)
        print(f"recorded {len(report.rows)} screen outcome(s) to {args.registry}")
    return 0


def cmd_holdout(args: argparse.Namespace) -> int:
    from .data.storage import CandleStore
    from .research import (
        ResearchRegistry,
        classify_holdout,
        holdout_test,
        setup_key,
        thresholds_from_config,
    )
    from .research.walkforward import DEFAULT_GRIDS

    config = _load_config(args.config)
    opt = config.optimize or {}
    store = CandleStore(args.data_dir)
    instruments = _instruments_from_args(config, args)
    candles_by_epic = {}
    for inst in instruments:
        bars = store.load(inst.epic, inst.timeframe)
        if not bars:
            log.error("no stored candles; run 'download' first", extra={"epic": inst.epic})
            return 2
        candles_by_epic[inst.epic] = bars

    name = args.strategy or config.strategy
    grid = (opt.get("param_grids", {}) or {}).get(name) or DEFAULT_GRIDS.get(name) \
        or opt.get("param_grid", {})
    if not grid:
        log.error("no parameter grid for strategy", extra={"strategy": name})
        return 2

    try:
        result = holdout_test(
            candles_by_epic, config, name, grid,
            holdout_frac=args.holdout_frac,
            metric=opt.get("metric", "sharpe"),
            warmup_bars=int(opt.get("warmup_bars", 250)),
            min_trades=int(opt.get("min_trades", 5)),
        )
    except ValueError as exc:
        log.error("holdout failed", extra={"error": str(exc)})
        return 2

    print("\n" + result.to_text())
    print("\nThis is your ONE clean test. If the holdout is positive with a real "
          "sample and survives cost-stress, it is worth a small demo forward-test. "
          "If it is flat/negative, the in-sample result was overfitting. Do not "
          "re-tune and re-run — that turns the holdout into just more snooping.\n")

    if args.record:
        # The holdout can only FAIL an idea or CLEAR it pending cost-stress. A
        # thin positive (too few trades or PF below the screen bar) is recorded as
        # an inconclusive `candidate`, NOT promoted — forward-test is a deliberate
        # decision made after cost-stress, never auto-granted here.
        status = classify_holdout(
            result.holdout_return_pct,
            result.holdout_profit_factor,
            result.holdout_trades,
            thresholds_from_config(config),
        )
        note = (f"auto: holdout {result.holdout_return_pct:.2f}% "
                f"PF {result.holdout_profit_factor:.2f} over {result.holdout_trades} trades "
                f"({result.holdout_start.date()}..{result.holdout_end.date()})")
        if status == "candidate":
            note += " — thin/inconclusive, run cost-stress before trusting"
        setup = setup_key(instruments)
        registry = ResearchRegistry.load(args.registry)
        registry.set_status(name, setup, status, note=note)
        registry.save(args.registry)
        print(f"recorded {name} @ {setup} -> {status} in {args.registry}")
        if status != "holdout-fail":
            print("NOTE: a positive holdout is NOT a green light. Run "
                  "`optimize --cost-stress` next; forward-test only after it survives.")
    return 0


def _format_registry(entries) -> str:
    lines = [
        "Research registry (strategy state per market setup):",
        f"  {'setup':<28} {'strategy':<20} {'status':<13} {'updated':<11} note",
    ]
    for e in entries:
        lines.append(
            f"  {e.setup:<28} {e.strategy:<20} {e.status:<13} {e.updated:<11} {e.note}"
        )
    return "\n".join(lines)


def cmd_registry(args: argparse.Namespace) -> int:
    from .research import ResearchRegistry, setup_key

    registry = ResearchRegistry.load(args.registry)

    if args.status:
        if not args.strategy:
            log.error("--status requires --strategy")
            return 2
        if args.setup:
            setup = args.setup
        elif getattr(args, "epics", None):
            tf = getattr(args, "timeframe", None) or "MINUTE_15"
            epics = [e.strip() for e in args.epics.split(",") if e.strip()]
            setup = setup_key(epics, timeframe=tf)
        else:
            log.error("need --setup or --epics (with --timeframe) to key the entry")
            return 2
        entry = registry.set_status(args.strategy, setup, args.status, note=args.note or "")
        registry.save(args.registry)
        print(f"recorded {entry.strategy} @ {entry.setup} -> {entry.status}")
        return 0

    entries = registry.entries()
    if not entries:
        print(f"registry is empty ({args.registry})")
        return 0
    print(_format_registry(entries))
    return 0


def _cost_stress_report(strategies, grid_for, run_wf, config) -> str:
    """Re-run walk-forward at increasing cost multiples; a real edge survives,
    a spread-driven artifact collapses as costs rise."""
    base = config.costs.spread_points
    multiples = [1.0, 2.0, 3.0, 5.0]
    lines = [
        f"Cost stress test (base spread = {base:g}; OOS return% / profit factor):",
        f"  {'strategy':<20} " + " ".join(f"{f'x{m:g}':>14}" for m in multiples),
    ]
    for name in strategies:
        grid = grid_for(name)
        if not grid:
            continue
        cells = []
        survived = True
        for m in multiples:
            try:
                r = run_wf(name, grid, cost_multiplier=m)
                cells.append(f"{r.combined_oos_return_pct:>6.2f}/{r.combined_profit_factor:<6.2f}")
                if r.combined_oos_return_pct <= 0 or r.combined_profit_factor <= 1.0:
                    survived = False
            except ValueError:
                cells.append(f"{'n/a':>13}")
                survived = False
        flag = "  <- holds up" if survived else ""
        lines.append(f"  {name:<20} " + " ".join(f"{c:>14}" for c in cells) + flag)
    lines.append("")
    lines.append("If a strategy turns negative (or PF <= 1) by 2-3x spread, the 'edge' was "
                 "spread-thin — not tradeable. Set costs.spread_points to your instrument's "
                 "REAL spread and trust that column.")
    return "\n".join(lines)


def _strategy_comparison(results, config=None) -> str:
    """Rank strategies by out-of-sample result and give a go/no-go verdict.

    Pass/fail uses the same mechanical rules as ``forex-bot report``
    (``research.verdicts``), so the two never diverge.
    """
    from .research import build_row, thresholds_from_config
    from .research.verdicts import VerdictThresholds

    if not results:
        return "No strategies could be evaluated (insufficient data or no grids)."
    thresholds = thresholds_from_config(config) if config is not None else VerdictThresholds()
    rows = sorted(
        (build_row(r, thresholds) for r in results),
        key=lambda x: x.oos_return_pct,
        reverse=True,
    )

    def _pf(pf: float) -> str:
        return "inf" if pf == float("inf") else f"{pf:.2f}"

    lines = [
        "Walk-forward comparison (out-of-sample, real data):",
        f"  {'strategy':<20} {'OOS ret%':>9} {'+folds%':>8} {'OOS PF':>7} "
        f"{'trades':>7} {'verdict':>8}",
    ]
    for r in rows:
        lines.append(
            f"  {r.strategy:<20} {r.oos_return_pct:>9.2f} "
            f"{r.pct_positive_folds:>8.0f} {_pf(r.profit_factor):>7} "
            f"{r.total_trades:>7} {r.verdict:>8}"
        )
    # PASS = cleared every rejection rule. "thin" = positive but failed a rule
    # (still likely noise, surfaced separately from outright duds).
    survivors = [r for r in rows if r.passed]
    thin = [
        r for r in rows
        if not r.passed and r.oos_return_pct > 0 and r.profit_factor > 1.0
    ]
    lines.append("")
    if survivors:
        names = ", ".join(r.strategy for r in survivors)
        lines.append(f"VERDICT: candidate(s) worth a closer look -> {names}")
    elif thin:
        names = ", ".join(r.strategy for r in thin)
        lines.append(f"VERDICT: marginal/thin positives ({names}) — likely noise, not edge.")
    else:
        lines.append("VERDICT: no strategy showed a robust out-of-sample edge on this data.")
        lines.append("Do NOT trade live. Try a higher timeframe (HOUR_4/DAY), more history, "
                     "or different instruments — or accept there is no tradeable edge here.")
    # These caveats apply to ANY positive result and are the usual reason a
    # "candidate" evaporates in live trading.
    if survivors or thin:
        lines.append("BEWARE before believing it:")
        lines.append("  * Multiple testing: many strategy/instrument/timeframe trials produce "
                     "false positives by chance. Count how many you have run.")
        lines.append("  * Costs: re-run with `optimize --cost-stress`; thin edges die under "
                     "realistic spreads (esp. wide CHF/cross spreads).")
        lines.append("  * Sample: a few dozen trades is not proof. Validate on more history "
                     "and an untouched out-of-sample period before risking capital.")
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


def cmd_spreads(args: argparse.Namespace) -> int:
    """Measure the current live bid/ask spread per instrument so cost
    assumptions are real numbers, not guesses."""
    config = _load_config(args.config)
    creds = CapitalCredentials.from_env()
    client = _make_client(creds, args)
    client.ensure_session()

    print(f"\n  {'epic':<12} {'bid':>12} {'offer':>12} {'spread_points':>14} {'config?':>9}")
    for inst in _instruments_from_args(config, args):
        try:
            details = client.get_market_details(inst.epic)
            snap = details.get("snapshot", {}) if isinstance(details, dict) else {}
            bid, offer = snap.get("bid"), snap.get("offer")
            if bid is None or offer is None:
                print(f"  {inst.epic:<12} {'?':>12} {'?':>12} {'unavailable':>14}")
                continue
            spread = float(offer) - float(bid)
            print(f"  {inst.epic:<12} {float(bid):>12.5f} {float(offer):>12.5f} "
                  f"{spread:>14.5f} {('set it' if spread > 0 else ''):>9}")
        except Exception as exc:
            print(f"  {inst.epic:<12} error: {exc}")
    print("\nPut these under each instrument as `spread_points:` in config.yaml so "
          "backtests price each instrument's real cost (vital for mixed baskets).\n")
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

    from .research.registry import STATUSES, ResearchRegistry

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
    o.add_argument("--cost-stress", action="store_true",
                   help="re-run at 1x-5x spread to see if an edge survives realistic costs")
    o.add_argument("--select-top", type=int,
                   help="per fold, keep only the top N instruments ranked on in-sample results")
    o.add_argument("--select-metric", choices=["return", "profit_factor", "expectancy", "trades"],
                   default="return", help="in-sample metric used by --select-top")
    _add_epics(o)
    o.set_defaults(func=cmd_optimize)

    h = sub.add_parser("holdout", help="optimize on early data, test ONCE on a held-out tail")
    h.add_argument("--strategy", help="strategy to test (default: config)")
    h.add_argument("--holdout-frac", type=float, default=0.2,
                   help="fraction of most-recent data reserved for the single test")
    h.add_argument("--registry", default=ResearchRegistry.DEFAULT_PATH,
                   help="research registry JSON path")
    h.add_argument("--record", action="store_true",
                   help="record the outcome (holdout-fail / forward-test) to the registry")
    _add_epics(h)
    h.set_defaults(func=cmd_holdout)

    rp = sub.add_parser(
        "report",
        help="rank every strategy under one walk-forward config with a PASS/FAIL verdict",
    )
    rp.add_argument("--strategy", help="evaluate only this strategy (default: all registered)")
    rp.add_argument("--out-dir", default="results",
                    help="directory for strategy_report.csv/.md (default: results/)")
    rp.add_argument("--select-top", type=int,
                    help="per fold, keep only the top N instruments ranked on in-sample results")
    rp.add_argument("--select-metric", choices=["return", "profit_factor", "expectancy", "trades"],
                    default="return", help="in-sample metric used by --select-top")
    rp.add_argument("--registry", default=ResearchRegistry.DEFAULT_PATH,
                    help="research registry JSON path (used to warn on frozen ideas)")
    rp.add_argument("--skip-frozen", action="store_true",
                    help="exclude strategies frozen (holdout-fail/retired) for this setup")
    rp.add_argument("--record", action="store_true",
                    help="record screen outcomes (candidate / screen-fail) to the registry")
    _add_epics(rp)
    rp.set_defaults(func=cmd_report)

    rg = sub.add_parser(
        "registry",
        help="view/update the research status registry (freeze failed ideas per setup)",
    )
    rg.add_argument("--registry", default=ResearchRegistry.DEFAULT_PATH,
                    help="research registry JSON path")
    rg.add_argument("--strategy", help="strategy to record (omit to just list the registry)")
    rg.add_argument("--status", choices=list(STATUSES),
                    help="status to record for --strategy on the given setup")
    rg.add_argument("--setup", help="explicit setup key (else built from --epics/--timeframe)")
    rg.add_argument("--note", help="freeform note attached to the entry")
    _add_epics(rg)
    rg.set_defaults(func=cmd_registry)

    se = sub.add_parser("search", help="search Capital.com for market epics by term")
    se.add_argument("term", help="search term, e.g. EURGBP or 'Australian Dollar'")
    se.set_defaults(func=cmd_search)

    sp = sub.add_parser("spreads", help="measure live bid/ask spread per instrument")
    _add_epics(sp)
    sp.set_defaults(func=cmd_spreads)

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
