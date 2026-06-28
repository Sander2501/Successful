"""Configuration and secret loading.

Two sources are merged:
  * Secrets / environment selection from process env (optionally a ``.env`` file).
  * Trading config (instruments, timeframes, risk params) from a YAML file.

Nothing here imports broker code, so it is safe to use everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # optional, only used if present
    import yaml
except Exception:  # pragma: no cover - yaml is a declared dep but stay defensive
    yaml = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# .env loading (tiny, dependency-free)
# --------------------------------------------------------------------------- #
def load_dotenv(path: str | Path = ".env") -> None:
    """Populate os.environ from a ``.env`` file if it exists.

    Intentionally minimal: ``KEY=VALUE`` lines, ``#`` comments, optional quotes.
    Existing environment variables are never overwritten.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #
@dataclass
class CapitalCredentials:
    environment: str  # "demo" | "live"
    identifier: str
    password: str
    api_key: str
    api_password: str | None = None
    demo_base_url: str = "https://demo-api-capital.backend-capital.com"
    live_base_url: str = "https://api-capital.backend-capital.com"
    demo_ws_url: str = "wss://api-streaming-capital.backend-capital.com/connect"
    live_ws_url: str = "wss://api-streaming-capital.backend-capital.com/connect"

    @property
    def is_live(self) -> bool:
        return self.environment.lower() == "live"

    @property
    def base_url(self) -> str:
        return self.live_base_url if self.is_live else self.demo_base_url

    @property
    def ws_url(self) -> str:
        return self.live_ws_url if self.is_live else self.demo_ws_url

    @classmethod
    def from_env(cls, *, load_env_file: bool = True) -> CapitalCredentials:
        if load_env_file:
            load_dotenv()
        env = os.environ
        missing = [
            k
            for k in ("CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD", "CAPITAL_API_KEY")
            if not env.get(k)
        ]
        if missing:
            raise RuntimeError(
                "Missing required Capital.com credentials in environment: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill it in."
            )
        return cls(
            environment=env.get("CAPITAL_ENVIRONMENT", "demo"),
            identifier=env["CAPITAL_IDENTIFIER"],
            password=env["CAPITAL_PASSWORD"],
            api_key=env["CAPITAL_API_KEY"],
            api_password=env.get("CAPITAL_API_PASSWORD") or None,
            demo_base_url=env.get(
                "CAPITAL_DEMO_BASE_URL", cls.demo_base_url
            ),
            live_base_url=env.get("CAPITAL_LIVE_BASE_URL", cls.live_base_url),
            demo_ws_url=env.get("CAPITAL_DEMO_WS_URL", cls.demo_ws_url),
            live_ws_url=env.get("CAPITAL_LIVE_WS_URL", cls.live_ws_url),
        )


# --------------------------------------------------------------------------- #
# trading / risk config
# --------------------------------------------------------------------------- #
@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01  # fraction of equity risked per trade
    max_position_pct: float = 0.20  # max notional fraction of equity per position
    max_open_positions: int = 5
    max_daily_loss_pct: float = 0.05  # halt new trades after this daily loss
    max_leverage: float = 30.0
    # Correlation / concentration controls. A single currency (e.g. USD) can be
    # the hidden common factor behind several "different" pairs, so cap net
    # exposure to any one currency and optionally the count of positions sharing
    # a currency. 1.0 / null effectively disable these checks.
    max_currency_exposure_pct: float = 1.0  # net per-currency notional / equity
    max_positions_per_currency: int | None = None
    # Data-driven correlation grouping (beyond shared currency codes). When
    # correlation_threshold is set, instruments whose return correlation exceeds
    # it are clustered, and net directional exposure within a cluster is capped.
    correlation_threshold: float | None = None  # e.g. 0.7; null disables
    max_correlated_exposure_pct: float = 1.0  # net group notional / equity
    max_positions_per_group: int | None = None
    # Re-estimate correlations every N closed candles in live trading (they drift
    # and can break in a crisis). null = estimate once from warmup only.
    correlation_refresh_bars: int | None = None
    # Portfolio kill switch: halt ALL new entries (and flatten) once equity falls
    # this far below its high-water mark. 1.0 effectively disables it.
    max_total_drawdown_pct: float = 1.0
    # Kill-switch peak window. null (default) = all-time high-water mark: the kill
    # switch is permanent until a manual reset, the most conservative behavior.
    # When set to N, drawdown is measured from the highest equity in the last N
    # equity observations, so an ancient peak expires and the switch becomes
    # *recoverable* — it re-arms once equity climbs back (hysteresis: it resumes
    # only after drawdown halves), letting the bot trade again on its own.
    drawdown_peak_window_bars: int | None = None
    # Position sizing: "fixed_fractional" (risk_per_trade via stop distance) or
    # "vol_target" (size so a 1-ATR move equals vol_target_pct of equity, which
    # equalizes risk contribution across instruments of different volatility).
    sizing_mode: str = "fixed_fractional"
    vol_target_pct: float = 0.01


@dataclass
class InstrumentConfig:
    epic: str
    timeframe: str = "MINUTE_15"
    value_per_point: float = 1.0  # account-ccy PnL per 1.0 price move per unit size
    # Per-instrument absolute spread in price units. Crucial for baskets that mix
    # price scales: a "pip" is 0.0001 on EUR/USD but 0.01 on USD/JPY, so a single
    # global spread mis-prices a mixed basket. null -> fall back to costs.spread_points.
    spread_points: float | None = None
    # Optional explicit FX decomposition; auto-parsed from a 6-letter epic
    # (e.g. "EURUSD" -> EUR/USD) when omitted.
    base_currency: str | None = None
    quote_currency: str | None = None


class InstrumentSpecs:
    """Per-instrument value-per-point and spread, with sensible fallbacks.

    Threaded through the portfolio, risk and execution layers so a basket that
    mixes price scales (EUR/USD ~1.1, USD/JPY ~150) is priced correctly instead
    of charging every instrument the first instrument's spread/value.
    """

    def __init__(self, instruments, *, default_spread: float = 0.0,
                 default_vpp: float = 1.0) -> None:
        self._vpp = {i.epic: i.value_per_point for i in instruments}
        self._spread = {
            i.epic: (i.spread_points if i.spread_points is not None else default_spread)
            for i in instruments
        }
        self.default_spread = default_spread
        self.default_vpp = default_vpp

    def vpp(self, epic: str) -> float:
        return self._vpp.get(epic, self.default_vpp)

    def spread(self, epic: str) -> float:
        return self._spread.get(epic, self.default_spread)


@dataclass
class CostConfig:
    """Simulated trading costs for the backtester."""

    spread_points: float = 0.0001  # absolute spread in price units
    commission_per_trade: float = 0.0
    slippage_points: float = 0.0


@dataclass
class TradingConfig:
    starting_equity: float = 10_000.0
    instruments: list[InstrumentConfig] = field(default_factory=list)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    strategy: str = "ema_crossover"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    # Walk-forward optimization settings (windows + parameter grid).
    optimize: dict[str, Any] = field(default_factory=dict)
    # SQLite file for durable live state (positions + risk high-water mark).
    # null disables persistence (state is kept only in memory).
    state_db: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TradingConfig:
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML config")
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradingConfig:
        instruments = [
            InstrumentConfig(**i) if isinstance(i, dict) else InstrumentConfig(epic=str(i))
            for i in data.get("instruments", [])
        ]
        risk = RiskConfig(**data.get("risk", {}))
        costs = CostConfig(**data.get("costs", {}))
        return cls(
            starting_equity=float(data.get("starting_equity", 10_000.0)),
            instruments=instruments,
            risk=risk,
            costs=costs,
            strategy=data.get("strategy", "ema_crossover"),
            strategy_params=data.get("strategy_params", {}),
            optimize=data.get("optimize", {}),
            state_db=data.get("state_db"),
        )
