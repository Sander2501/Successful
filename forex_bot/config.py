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
from typing import Any, Optional

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
    api_password: Optional[str] = None
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
    def from_env(cls, *, load_env_file: bool = True) -> "CapitalCredentials":
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


@dataclass
class InstrumentConfig:
    epic: str
    timeframe: str = "MINUTE_15"
    value_per_point: float = 1.0  # account-ccy PnL per 1.0 price move per unit size


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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TradingConfig":
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML config")
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradingConfig":
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
        )
