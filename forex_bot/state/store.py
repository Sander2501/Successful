"""SQLite-backed persistence for live trading state.

The plan requires the bot to be "restartable without losing knowledge of
positions". This store persists the two things a restart must not forget:

  * open positions (with their protective levels and broker deal ids), and
  * risk state — the equity high-water mark, the kill-switch flag, and the
    daily-loss bookkeeping.

Without it, a crash-and-restart resets the drawdown high-water mark, so a bot
that had already tripped (or was close to) its kill switch would happily resume
trading. Pure stdlib (`sqlite3`); safe to use anywhere.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from ..models import Position, Side

_POSITION_COLUMNS = (
    "epic", "side", "size", "entry_price",
    "deal_id", "stop_loss", "take_profit", "opened_at",
)


class StateStore:
    """A tiny persistence layer over SQLite. Usable as a context manager."""

    def __init__(self, path: str | Path = "data/db/state.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS positions (
                epic        TEXT PRIMARY KEY,
                side        TEXT NOT NULL,
                size        REAL NOT NULL,
                entry_price REAL NOT NULL,
                deal_id     TEXT,
                stop_loss   REAL,
                take_profit REAL,
                opened_at   TEXT
            );
            CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # positions
    # ------------------------------------------------------------------ #
    def save_positions(self, positions) -> None:
        """Replace the stored position set with ``positions`` (full snapshot)."""
        with self._conn:
            self._conn.execute("DELETE FROM positions")
            self._conn.executemany(
                f"INSERT INTO positions ({','.join(_POSITION_COLUMNS)}) "
                f"VALUES (?,?,?,?,?,?,?,?)",
                [self._position_row(p) for p in positions],
            )

    def upsert_position(self, position: Position) -> None:
        with self._conn:
            self._conn.execute(
                f"INSERT INTO positions ({','.join(_POSITION_COLUMNS)}) "
                f"VALUES (?,?,?,?,?,?,?,?) "
                f"ON CONFLICT(epic) DO UPDATE SET "
                f"side=excluded.side, size=excluded.size, "
                f"entry_price=excluded.entry_price, deal_id=excluded.deal_id, "
                f"stop_loss=excluded.stop_loss, take_profit=excluded.take_profit, "
                f"opened_at=excluded.opened_at",
                self._position_row(position),
            )

    def remove_position(self, epic: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM positions WHERE epic = ?", (epic,))

    def load_positions(self) -> list[Position]:
        rows = self._conn.execute("SELECT * FROM positions").fetchall()
        return [self._row_to_position(r) for r in rows]

    # ------------------------------------------------------------------ #
    # key/value risk state
    # ------------------------------------------------------------------ #
    def set_value(self, key: str, value: Any) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def get_value(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def save_risk_state(self, state: dict) -> None:
        self.set_value("risk_state", state)

    def load_risk_state(self) -> dict:
        return self.get_value("risk_state", {}) or {}

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _position_row(p: Position) -> tuple:
        return (
            p.epic, p.side.value, p.size, p.entry_price,
            p.deal_id, p.stop_loss, p.take_profit,
            p.opened_at.isoformat() if p.opened_at else None,
        )

    @staticmethod
    def _row_to_position(r: sqlite3.Row) -> Position:
        return Position(
            epic=r["epic"],
            side=Side(r["side"]),
            size=r["size"],
            entry_price=r["entry_price"],
            deal_id=r["deal_id"],
            stop_loss=r["stop_loss"],
            take_profit=r["take_profit"],
            opened_at=datetime.fromisoformat(r["opened_at"]) if r["opened_at"] else None,
        )
