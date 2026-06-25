"""Data-driven correlation grouping for risk concentration limits.

The per-currency caps in :mod:`exposure` catch instruments that share a literal
currency code. But correlation is broader: EUR/USD and the dollar index move
together without sharing a ticker substring, and some pairs are *negatively*
correlated. This module estimates pairwise return correlation from candle data
and clusters instruments into correlated groups so the risk manager can cap
net directional exposure within a group, sign-aware.

Pure stdlib; correlations are computed once per backtest (or from warmup history
in live trading) and handed to the :class:`~forex_bot.risk.manager.RiskManager`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..models import Candle


def aligned_returns(candles_by_epic: dict[str, list[Candle]]) -> dict[str, list[float]]:
    """Return per-epic return series aligned on common timestamps.

    Only timestamps present for *every* epic are used, so the series line up
    index-for-index and correlations are computed on matching bars.
    """
    if len(candles_by_epic) < 2:
        return {epic: [] for epic in candles_by_epic}

    closes_by_ts = {
        epic: {c.timestamp: c.close for c in candles}
        for epic, candles in candles_by_epic.items()
    }
    common = None
    for ts_map in closes_by_ts.values():
        keys = set(ts_map)
        common = keys if common is None else (common & keys)
    common_sorted = sorted(common or set())

    out: dict[str, list[float]] = {}
    for epic, ts_map in closes_by_ts.items():
        series = [ts_map[ts] for ts in common_sorted]
        rets = [
            (cur / prev - 1.0) if prev else 0.0
            for prev, cur in zip(series, series[1:])
        ]
        out[epic] = rets
    return out


def pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / math.sqrt(va * vb)


@dataclass
class CorrelationModel:
    """Correlation matrix + clusters with a signed reference per group."""

    matrix: dict[tuple[str, str], float] = field(default_factory=dict)
    _group_of: dict[str, int] = field(default_factory=dict)
    _members: dict[int, set[str]] = field(default_factory=dict)
    _sign: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_candles(
        cls, candles_by_epic: dict[str, list[Candle]], threshold: float
    ) -> "CorrelationModel":
        epics = list(candles_by_epic)
        rets = aligned_returns(candles_by_epic)
        matrix: dict[tuple[str, str], float] = {}
        for i, a in enumerate(epics):
            for b in epics[i + 1 :]:
                c = pearson(rets.get(a, []), rets.get(b, []))
                matrix[(a, b)] = c
                matrix[(b, a)] = c

        # Union-find clustering on |corr| >= threshold.
        parent = {e: e for e in epics}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for (a, b), c in matrix.items():
            if a < b and abs(c) >= threshold:
                union(a, b)

        # Assign group ids and a sign relative to each group's reference epic.
        roots = {e: find(e) for e in epics}
        group_ids: dict[str, int] = {}
        members: dict[int, set[str]] = {}
        next_id = 0
        root_to_id: dict[str, int] = {}
        for e in epics:
            r = roots[e]
            if r not in root_to_id:
                root_to_id[r] = next_id
                next_id += 1
            gid = root_to_id[r]
            group_ids[e] = gid
            members.setdefault(gid, set()).add(e)

        sign: dict[str, int] = {}
        for gid, group in members.items():
            ref = min(group)  # deterministic reference
            for e in group:
                if e == ref:
                    sign[e] = 1
                else:
                    sign[e] = 1 if matrix.get((e, ref), 0.0) >= 0 else -1
        return cls(matrix=matrix, _group_of=group_ids, _members=members, _sign=sign)

    def group_of(self, epic: str) -> Optional[int]:
        return self._group_of.get(epic)

    def group_members(self, epic: str) -> set[str]:
        gid = self._group_of.get(epic)
        return set(self._members.get(gid, set())) if gid is not None else {epic}

    def sign(self, epic: str) -> int:
        return self._sign.get(epic, 1)

    def correlation(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        return self.matrix.get((a, b), 0.0)

    def has_groups(self) -> bool:
        """True if at least one group has more than one member."""
        return any(len(m) > 1 for m in self._members.values())
