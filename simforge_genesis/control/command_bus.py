"""Priority command bus for the controller."""
from __future__ import annotations

import asyncio
from typing import Dict, Iterable, Optional

from ..core import Command, CommandPriority, CommandRejected, CommandAccepted, EventTopic
from ..core.commands import StopCommand
from .event_bus import EventBus


class CommandBus:
    """Queues robot commands with priority ordering."""

    def __init__(self, event_bus: EventBus, known_robots: Iterable[str]) -> None:
        self._queues: Dict[str, asyncio.PriorityQueue[tuple[int, int, Command]]] = {
            name: asyncio.PriorityQueue() for name in known_robots
        }
        self._counter = 0
        self._event_bus = event_bus
        self._robots = set(known_robots)
        self._stopped = asyncio.Event()

    async def submit(self, command: Command) -> None:
        if command.robot not in self._robots:
            await self._event_bus.publish(
                CommandRejected(
                    topic=EventTopic.COMMAND,
                    robot=command.robot,
                    command_name=type(command).__name__,
                    reason="unknown_robot",
                )
            )
            return
        if isinstance(command, StopCommand):
            self._stopped.set()
        priority = int(command.priority)
        self._counter += 1
        queue = self._queues[command.robot]
        await queue.put((priority, self._counter, command))
        await self._event_bus.publish(
            CommandAccepted(
                topic=EventTopic.COMMAND,
                robot=command.robot,
                command_name=type(command).__name__,
                priority=command.priority,
            )
        )

    async def get(self, robot: str) -> Command:
        if robot not in self._queues:
            raise KeyError(f"Unknown robot '{robot}'")
        priority, _, command = await self._queues[robot].get()
        return command

    def pending_count(self, robot: str) -> int:
        queue = self._queues.get(robot)
        if queue is None:
            raise KeyError(f"Unknown robot '{robot}'")
        return queue.qsize()

    async def stop_all(self, reason: str = "shutdown") -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        for robot in self._robots:
            await self.submit(
                StopCommand(
                    robot_name=robot,
                    priority=CommandPriority.EMERGENCY_STOP,
                    metadata={"reason": reason},
                )
            )

    def is_stopped(self) -> bool:
        return self._stopped.is_set()


__all__ = ["CommandBus"]
