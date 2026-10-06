"""Small deterministic progress reporter shared by live benchmark runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AttemptProgress:
    """Report initial, roughly one-percent, and final processed-attempt counts."""

    label: str
    total: int
    completed: int = 0
    callback: Callable[[str], None] | None = None
    _last_reported: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        self.total = max(0, int(self.total))
        self.completed = min(max(0, int(self.completed)), self.total)
        self._emit(force=True)

    def advance(self, count: int = 1) -> None:
        self.completed = min(self.total, self.completed + max(0, int(count)))
        self._emit(force=self.completed == self.total)

    def _emit(self, *, force: bool) -> None:
        if self.callback is None or self.completed == self._last_reported:
            return
        interval = max(1, self.total // 100)
        if not force and self.completed % interval:
            return
        percent = 100.0 if self.total == 0 else 100.0 * self.completed / self.total
        self.callback(
            f"{self.label}: {self.completed}/{self.total} attempts processed ({percent:.1f}%)"
        )
        self._last_reported = self.completed


__all__ = ["AttemptProgress"]
