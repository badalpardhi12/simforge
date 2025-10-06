"""Simple publish/subscribe event bus."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable, Dict, List

from ..core import Event

Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Async event bus used by the orchestration layer."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Handler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(self, event: Event) -> None:
        key = event.topic.value if hasattr(event.topic, "value") else str(event.topic)
        async with self._lock:
            subscribers = list(self._subscribers.get(key, [])) + list(self._subscribers.get("*", []))
        pending = []
        for handler in subscribers:
            result = handler(event)
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                pending.append(asyncio.ensure_future(result))
        if pending:
            await asyncio.gather(*pending)

    def subscribe(self, topic: str, handler: Handler):
        async def _register() -> None:
            async with self._lock:
                self._subscribers[topic].append(handler)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_register())
            return None
        else:
            return loop.create_task(_register())

    def emit(self, event: Event) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.publish(event))
        else:
            loop.create_task(self.publish(event))

    def unsubscribe(self, topic: str, handler: Handler):
        async def _remove() -> None:
            async with self._lock:
                handlers = self._subscribers.get(topic)
                if handlers and handler in handlers:
                    handlers.remove(handler)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_remove())
            return None
        else:
            return loop.create_task(_remove())


__all__ = ["EventBus"]
