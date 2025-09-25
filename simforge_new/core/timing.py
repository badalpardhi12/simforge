"""Time abstractions and helpers for coordinating simulation + wall time."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


class TimeSource(Protocol):
    """Minimal protocol implemented by clock-like objects."""

    def now(self) -> float:  # pragma: no cover - protocol definition
        """Return the current timestamp in seconds."""


class WallClock(TimeSource):
    """Real-time clock backed by :func:`time.time`."""

    def now(self) -> float:
        return time.time()


@dataclass(frozen=True)
class SimulationClock(TimeSource):
    """Deterministic clock that advances in discrete steps."""

    dt: float
    current_step: int = 0

    def now(self) -> float:
        return self.current_step * self.dt

    def advance(self, steps: int = 1) -> "SimulationClock":
        """Return a new clock advanced by ``steps`` simulation ticks."""

        return SimulationClock(dt=self.dt, current_step=self.current_step + steps)


@dataclass
class Timer:
    """Stopwatch-style helper for measuring elapsed time.

    The timer can be reused multiple times and supports context-manager usage::

        with Timer() as t:
            do_work()
        print(t.elapsed)
    """

    clock: TimeSource = field(default_factory=WallClock)
    _start: float | None = field(init=False, default=None)
    _accumulated: float = field(init=False, default=0.0)

    def start(self) -> None:
        if self._start is None:
            self._start = self.clock.now()

    def stop(self) -> None:
        if self._start is not None:
            self._accumulated += self.clock.now() - self._start
            self._start = None

    def reset(self) -> None:
        self._start = None
        self._accumulated = 0.0

    @property
    def running(self) -> bool:
        return self._start is not None

    @property
    def elapsed(self) -> float:
        total = self._accumulated
        if self._start is not None:
            total += self.clock.now() - self._start
        return total

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


@dataclass
class RateController:
    """Utility that sleeps to maintain a target loop frequency."""

    frequency_hz: float
    clock: TimeSource = field(default_factory=WallClock)
    _last_tick: float = field(init=False)

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        self._period = 1.0 / self.frequency_hz
        self._last_tick = self.clock.now()

    def sleep(self) -> float:
        """Block until the next tick.

        Returns
        -------
        float
            Actual sleep duration in seconds (may be zero or negative if running behind).
        """

        target = self._last_tick + self._period
        now = self.clock.now()
        remaining = target - now

        if remaining > 0 and isinstance(self.clock, WallClock):
            time.sleep(remaining)
            now = target
        else:
            # For non-wall clocks or if we're lagging behind, advance immediately.
            now = max(target, self.clock.now())

        self._last_tick = now
        return remaining


@dataclass
class TimeSync:
    """Translate timestamps between simulation and wall clocks."""

    sim_clock: TimeSource
    wall_clock: TimeSource = field(default_factory=WallClock)
    _anchor_wall: float = field(init=False)
    _anchor_sim: float = field(init=False)

    def __post_init__(self) -> None:
        self._anchor_wall = self.wall_clock.now()
        self._anchor_sim = self.sim_clock.now()

    def wall_time(self) -> float:
        return self.wall_clock.now()

    def sim_time(self) -> float:
        return self.sim_clock.now()

    def to_wall(self, sim_time: float) -> float:
        return self._anchor_wall + (sim_time - self._anchor_sim)

    def to_sim(self, wall_time: float) -> float:
        return self._anchor_sim + (wall_time - self._anchor_wall)

    def refresh(self) -> None:
        self._anchor_wall = self.wall_clock.now()
        self._anchor_sim = self.sim_clock.now()


__all__ = [
    "TimeSource",
    "WallClock",
    "SimulationClock",
    "Timer",
    "RateController",
    "TimeSync",
]
