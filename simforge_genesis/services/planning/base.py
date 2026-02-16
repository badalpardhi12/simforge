"""Motion planning service contracts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Protocol, Tuple

from ...core import PlannerStrategy
from ...core.models import RobotProfile
from ...core.trajectories import JointTrajectory

JointVector = Tuple[float, ...]


class PlanOutcome(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    COLLISION = "collision"
    INVALID_GOAL = "invalid_goal"
    FAILURE = "failure"


@dataclass(frozen=True)
class PlanRequest:
    robot: RobotProfile
    start: JointVector
    goal: JointVector
    strategy: PlannerStrategy
    timeout_s: float = 3.0
    allow_partial: bool = False
    is_state_valid: Optional[Callable[[JointVector], bool]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", tuple(float(v) for v in self.start))
        object.__setattr__(self, "goal", tuple(float(v) for v in self.goal))


@dataclass(frozen=True)
class PlannerMetrics:
    planner_name: str
    duration_s: float
    states_validated: int
    clearance_min_m: Optional[float] = None
    diagnostics: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.diagnostics is None:
            object.__setattr__(self, "diagnostics", {})


@dataclass(frozen=True)
class PlanResult:
    outcome: PlanOutcome
    trajectory: Optional[JointTrajectory]
    metrics: Optional[PlannerMetrics] = None
    raw: Dict[str, object] = None

    def __post_init__(self) -> None:
        if self.raw is None:
            object.__setattr__(self, "raw", {})


class MotionPlanner(Protocol):
    """Protocol implemented by motion planners (OMPL, linear, etc.)."""

    def plan(self, request: PlanRequest) -> PlanResult:  # pragma: no cover
        ...


__all__ = [
    "PlanOutcome",
    "PlanRequest",
    "PlannerMetrics",
    "PlanResult",
    "MotionPlanner",
]
