"""Research registry: the discipline layer that freezes failed ideas.

A strategy's research state is tracked *per market setup* — a (timeframe,
instrument-universe) pair — because an edge that dies on MINUTE_15 EUR/GBP/USD
crosses has not been tested on HOUR_4, and should not be treated as if it had.

Lifecycle states:
    candidate     passed the walk-forward screen (or a thin/inconclusive holdout);
                  not yet a proven clean pass
    screen-fail   failed the walk-forward screen (may be re-screened with a
                  different grid — the screen is not the clean test)
    holdout-fail  failed the ONE clean out-of-sample holdout (FROZEN — re-running
                  the same period with new params is data-snooping)
    holdout-pass  cleared the holdout with an adequate sample and margin; run
                  cost-stress next, then decide on a demo forward-test
    forward-test  in demo forward-testing (a deliberate post-cost-stress decision)
    live          traded with real capital
    retired       deliberately shelved (FROZEN)

"Frozen" setups (``holdout-fail`` / ``retired``) are the point of this module:
they stop you from accidentally re-opening a spent idea on the same setup.

The registry is a small JSON file that is meant to be **committed** (unlike the
gitignored ``results/`` artifacts) so the research record is shared and durable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

# Ordered loosely by progression; used only for validation, not ranking.
STATUSES = (
    "candidate",
    "screen-fail",
    "holdout-fail",
    "holdout-pass",
    "forward-test",
    "live",
    "retired",
)

# Terminal states: an automated screen run must not silently overwrite these,
# and they freeze the (strategy, setup) against re-opening.
FROZEN_STATUSES = frozenset({"holdout-fail", "retired"})

# The only statuses a demo/live engine may launch with (see cli._frozen_setup_check).
# screen-fail is deliberately NOT launchable: it failed the walk-forward screen.
ALLOWED_LAUNCH_STATUSES = frozenset({"candidate", "holdout-pass", "forward-test", "live"})


def setup_key(instruments: Sequence, *, timeframe: str | None = None) -> str:
    """Canonical key for a market setup, e.g. ``HOUR_4:AUDUSD,EURUSD,GBPUSD``.

    Accepts InstrumentConfig-like objects (with ``.epic`` / ``.timeframe``) or
    plain epic strings (then ``timeframe`` must be given). Epics are sorted so the
    key is order-independent.
    """
    epics: list[str] = []
    timeframes: set[str] = set()
    for inst in instruments:
        if isinstance(inst, str):
            epics.append(inst)
        else:
            epics.append(inst.epic)
            tf = getattr(inst, "timeframe", None)
            if tf:
                timeframes.add(tf)
    if timeframe:
        timeframes = {timeframe}
    tf_label = "/".join(sorted(timeframes)) if timeframes else "?"
    return f"{tf_label}:{','.join(sorted(epics))}"


@dataclass
class RegistryEntry:
    strategy: str
    setup: str
    status: str
    note: str = ""
    updated: str = field(default_factory=lambda: date.today().isoformat())

    @property
    def frozen(self) -> bool:
        return self.status in FROZEN_STATUSES


def _entry_id(strategy: str, setup: str) -> str:
    return f"{strategy}@{setup}"


class ResearchRegistry:
    """A JSON-backed ledger of strategy research state, keyed by (strategy, setup)."""

    DEFAULT_PATH = "research/registry.json"

    def __init__(self, entries: dict[str, RegistryEntry] | None = None,
                 path: str | Path | None = None):
        self._entries: dict[str, RegistryEntry] = entries or {}
        self.path = Path(path) if path else None

    # ---- persistence ---------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path) -> ResearchRegistry:
        p = Path(path)
        if not p.exists():
            return cls(entries={}, path=p)
        raw = json.loads(p.read_text() or "{}")
        entries = {
            _entry_id(e["strategy"], e["setup"]): RegistryEntry(**e)
            for e in raw.get("entries", [])
        }
        return cls(entries=entries, path=p)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path to save the registry to")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [asdict(e) for e in self._sorted()]}
        target.write_text(json.dumps(payload, indent=2) + "\n")
        self.path = target
        return target

    # ---- queries -------------------------------------------------------- #
    def get(self, strategy: str, setup: str) -> RegistryEntry | None:
        return self._entries.get(_entry_id(strategy, setup))

    def is_frozen(self, strategy: str, setup: str) -> bool:
        entry = self.get(strategy, setup)
        return bool(entry and entry.frozen)

    def entries(self) -> list[RegistryEntry]:
        return self._sorted()

    def _sorted(self) -> list[RegistryEntry]:
        return sorted(self._entries.values(), key=lambda e: (e.setup, e.strategy))

    # ---- mutation ------------------------------------------------------- #
    def set_status(self, strategy: str, setup: str, status: str, *,
                   note: str = "", allow_overwrite_frozen: bool = True) -> RegistryEntry:
        """Record a status. Raises on an unknown status.

        When ``allow_overwrite_frozen`` is False, an existing FROZEN entry is left
        untouched (used by automated screen recording so a screen pass cannot
        silently re-open a holdout-fail). Returns the effective entry.
        """
        if status not in STATUSES:
            raise ValueError(f"unknown status '{status}'; one of {list(STATUSES)}")
        existing = self.get(strategy, setup)
        if existing and existing.frozen and not allow_overwrite_frozen:
            return existing
        entry = RegistryEntry(strategy=strategy, setup=setup, status=status, note=note)
        self._entries[_entry_id(strategy, setup)] = entry
        return entry
