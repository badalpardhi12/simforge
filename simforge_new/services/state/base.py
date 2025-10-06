"""Robot state estimation service contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Tuple

from ...core.models import RobotProfile, RobotState

JointVector = Tuple[float, ...]


@dataclass(frozen=True)
class StateSubscription:
    robot: RobotProfile
    frequency_hz: float = 60.0


class StateEstimator(Protocol):
    """Protocol publishing robot joint states."""

    async def states(self, subscription: StateSubscription) -> AsyncIterator[RobotState]:  # pragma: no cover
        ...

    async def set_reference(self, robot: RobotProfile, joints: JointVector) -> None:  # pragma: no cover
        ...


__all__ = ["StateSubscription", "StateEstimator"]
