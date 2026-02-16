"""Event system for SimForge."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .commands import CommandPriority
from .models import RobotState


class EventTopic(str, Enum):
    """High-level topics used for the event bus."""

    COMMAND = "command"
    ROBOT_STATE = "robot_state"
    TRAJECTORY = "trajectory"
    SYSTEM = "system"
    TELEMETRY = "telemetry"
    CUSTOM = "custom"


class EventType(Enum):
    """Legacy enum retained for compatibility."""

    COMMAND = EventTopic.COMMAND.value
    ROBOT_STATE = EventTopic.ROBOT_STATE.value
    SYSTEM = EventTopic.SYSTEM.value
    CUSTOM = EventTopic.CUSTOM.value


@dataclass(frozen=True)
class Event:
    """Base event published on the event bus."""

    topic: EventTopic | str
    robot: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp_s: float = field(default_factory=lambda: time.time())
    source: Optional[str] = None

    @property
    def event_type(self) -> str:
        return self.topic.value if isinstance(self.topic, EventTopic) else str(self.topic)


@dataclass(frozen=True)
class CommandEvent(Event):
    command_name: str = ""
    command_id: Optional[str] = None

    def __post_init__(self) -> None:
        data = dict(self.payload)
        if self.command_name:
            data.setdefault("command_name", self.command_name)
        if self.command_id is not None:
            data.setdefault("command_id", self.command_id)
        object.__setattr__(self, "payload", data)


@dataclass(frozen=True)
class CommandAccepted(CommandEvent):
    priority: CommandPriority = CommandPriority.NORMAL

    def __post_init__(self) -> None:
        super().__post_init__()
        data = dict(self.payload)
        data.setdefault("priority", self.priority)
        object.__setattr__(self, "payload", data)


@dataclass(frozen=True)
class CommandRejected(CommandEvent):
    reason: str = "unknown"

    def __post_init__(self) -> None:
        super().__post_init__()
        data = dict(self.payload)
        data.setdefault("reason", self.reason)
        object.__setattr__(self, "payload", data)


@dataclass(frozen=True, kw_only=True)
class RobotStateSnapshot(Event):
    state: RobotState

    def __post_init__(self) -> None:
        data = dict(self.payload)
        data.setdefault("joint_positions", self.state.joint_positions)
        data.setdefault("timestamp_s", self.state.timestamp_s)
        object.__setattr__(self, "payload", data)
        if not self.robot:
            object.__setattr__(self, "robot", self.state.name)


@dataclass(frozen=True)
class ErrorEvent(Event):
    topic: EventTopic = EventTopic.SYSTEM
    message: str = ""
    recoverable: bool = True

    def __post_init__(self) -> None:
        data = dict(self.payload)
        if self.message:
            data.setdefault("message", self.message)
        data.setdefault("recoverable", self.recoverable)
        object.__setattr__(self, "payload", data)


EventHandler = Callable[[Event], None]


@dataclass
class EventSubscription:
    """Represents a subscription to events."""

    topic: str
    handler: EventHandler
    filter: Optional[Callable[[Event], bool]] = None
    subscription_id: str = field(default_factory=lambda: str(time.time()))
    active: bool = True

    def should_handle(self, event: Event) -> bool:
        if not self.active:
            return False
        if self.topic != "*" and event.topic.value != self.topic:
            return False
        if self.filter and not self.filter(event):
            return False
        return True

    def handle(self, event: Event) -> None:
        if self.should_handle(event):
            self.handler(event)


__all__ = [
    "EventTopic",
    "EventType",
    "Event",
    "CommandEvent",
    "CommandAccepted",
    "CommandRejected",
    "RobotStateSnapshot",
    "ErrorEvent",
    "EventHandler",
    "EventSubscription",
]
