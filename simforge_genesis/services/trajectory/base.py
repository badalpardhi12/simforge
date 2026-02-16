"""Trajectory execution service contracts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Protocol

from ...core.trajectories import JointTrajectory


class ExecutionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class ExecutionHandle:
    trajectory_id: str
    status: ExecutionStatus
    metadata: Dict[str, object]


class TrajectoryExecutor(Protocol):
    """Protocol that streams joint trajectories to the simulator or hardware."""

    async def start(self, trajectory: JointTrajectory) -> ExecutionHandle:  # pragma: no cover
        ...

    async def stop(self, trajectory_id: str, reason: Optional[str] = None) -> ExecutionHandle:  # pragma: no cover
        ...

    async def query(self, trajectory_id: str) -> ExecutionHandle:  # pragma: no cover
        ...


__all__ = ["ExecutionStatus", "ExecutionHandle", "TrajectoryExecutor"]
