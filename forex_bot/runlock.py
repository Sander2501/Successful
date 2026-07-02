"""Single-instance run lock for the demo/live engine.

Two engines trading the same account and instruments double every position and
silently bypass every risk cap (each process sees only its own positions), so
``forex-bot demo``/``live`` acquire an exclusive PID lockfile before starting.

A lock left behind by a crashed process (its PID no longer alive) is treated as
stale and taken over, so a crash never requires manual cleanup.
"""

from __future__ import annotations

import os
from pathlib import Path

from .logging_setup import get_logger

log = get_logger(__name__)


class AlreadyRunningError(RuntimeError):
    """Another live/demo engine holds the lock (its process is alive)."""

    def __init__(self, path: Path, pid: int) -> None:
        super().__init__(
            f"another engine appears to be running (pid {pid}); "
            f"stop it first, or remove {path} if this is wrong"
        )
        self.path = path
        self.pid = pid


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0: existence check only, nothing is delivered
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False
    return True


class RunLock:
    """PID lockfile with stale-lock takeover. Use as a context manager."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._acquired = False

    def acquire(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # O_EXCL create: atomic when no lock exists (the common case).
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid = self._read_pid()
            if pid and pid != os.getpid() and _pid_alive(pid):
                raise AlreadyRunningError(self.path, pid) from None
            log.warning("stale run lock found; taking over",
                        extra={"path": str(self.path), "stale_pid": pid})
            self.path.write_text(str(os.getpid()))
        else:
            with os.fdopen(fd, "w") as fh:
                fh.write(str(os.getpid()))
        self._acquired = True
        return self

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            if self._read_pid() == os.getpid():  # never delete someone else's lock
                self.path.unlink()
        except OSError:
            pass
        self._acquired = False

    def _read_pid(self) -> int:
        try:
            return int(self.path.read_text().strip() or 0)
        except (OSError, ValueError):
            return 0

    def __enter__(self) -> RunLock:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
