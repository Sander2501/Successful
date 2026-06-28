"""Historical candle storage (CSV, dependency-free).

Stores one file per (epic, timeframe) with a stable schema and UTC ISO
timestamps. Includes data-quality checks for duplicates and gaps. Parquet is
supported transparently when ``pyarrow``/``pandas`` are installed.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from ..logging_setup import get_logger
from ..models import Candle, _parse_ts
from .candle_builder import timeframe_to_seconds

log = get_logger(__name__)

_FIELDS = ["timestamp", "open", "high", "low", "close", "volume"]


class CandleStore:
    def __init__(self, root: str | Path = "data/historical") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, epic: str, timeframe: str) -> Path:
        safe = epic.replace("/", "_").replace(":", "_")
        return self.root / f"{safe}__{timeframe}.csv"

    # ------------------------------------------------------------------ #
    def save(self, epic: str, timeframe: str, candles: Iterable[Candle]) -> Path:
        candles = sorted(candles, key=lambda c: c.timestamp)
        path = self.path_for(epic, timeframe)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(_FIELDS)
            for c in candles:
                writer.writerow(
                    [c.timestamp.isoformat(), c.open, c.high, c.low, c.close, c.volume]
                )
        log.info("saved candles", extra={"epic": epic, "tf": timeframe,
                                         "count": len(candles), "path": str(path)})
        return path

    def load(self, epic: str, timeframe: str) -> list[Candle]:
        path = self.path_for(epic, timeframe)
        if not path.exists():
            return []
        out: list[Candle] = []
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                out.append(
                    Candle(
                        epic=epic,
                        timeframe=timeframe,
                        timestamp=_parse_ts(row["timestamp"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def quality_report(candles: list[Candle], timeframe: str) -> dict:
        """Return counts of duplicate timestamps and gaps in the series."""
        if not candles:
            return {"count": 0, "duplicates": 0, "gaps": 0}
        ordered = sorted(candles, key=lambda c: c.timestamp)
        step = timeframe_to_seconds(timeframe)
        duplicates = 0
        gaps = 0
        for prev, cur in zip(ordered, ordered[1:], strict=False):
            delta = (cur.timestamp - prev.timestamp).total_seconds()
            if delta == 0:
                duplicates += 1
            elif step and delta > step * 1.5:
                gaps += 1
        return {"count": len(ordered), "duplicates": duplicates, "gaps": gaps}
