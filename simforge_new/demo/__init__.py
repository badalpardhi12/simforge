"""Built-in demo scenarios for Simforge."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict

from . import face_robot

DemoCoroutine = Callable[[], Awaitable[None]]

_DEMOS: Dict[str, tuple[DemoCoroutine, str]] = {
    "face_robot": (face_robot.run_demo, face_robot.DESCRIPTION),
}


def available_demos() -> Dict[str, str]:
    """Return mapping of demo names to descriptions."""
    return {name: meta[1] for name, meta in _DEMOS.items()}


def ensure_demo(name: str) -> DemoCoroutine:
    try:
        return _DEMOS[name][0]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown demo '{name}'. Available demos: {', '.join(sorted(_DEMOS))}."
        ) from exc


def run_demo(name: str) -> None:
    """Execute the named demo synchronously."""
    demo = ensure_demo(name)
    asyncio.run(demo())


__all__ = ["available_demos", "run_demo", "ensure_demo", "DemoCoroutine"]
