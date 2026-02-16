"""Inverse kinematics service contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple

from ...core import MotionTarget
from ...core.models import RobotProfile

JointVector = Tuple[float, ...]


@dataclass(frozen=True)
class IKRequest:
    robot: RobotProfile
    target: MotionTarget
    seed: Optional[JointVector] = None
    max_attempts: int = 1
    timeout_s: float = 0.5
    prefer_cartesian: bool = True
    position_tolerance_m: Optional[float] = None
    orientation_tolerance_deg: Optional[float] = None
    is_state_valid: Optional[Callable[[JointVector], bool]] = None


@dataclass(frozen=True)
class IKMetrics:
    position_error_m: float
    orientation_error_deg: float
    attempts: int
    slack_used: bool = False
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class IKResult:
    success: bool
    solution: Optional[JointVector]
    metrics: Optional[IKMetrics] = None
    raw: Dict[str, object] = None

    def __post_init__(self) -> None:
        if self.solution is not None:
            object.__setattr__(self, "solution", tuple(float(v) for v in self.solution))
        if self.metrics is None:
            object.__setattr__(self, "metrics", None)
        if self.raw is None:
            object.__setattr__(self, "raw", {})


class IKSolver(Protocol):
    """Protocol implemented by IK backends (Drake, analytic, etc.)."""

    def solve(self, request: IKRequest) -> IKResult:  # pragma: no cover - interface definition
        ...


__all__ = ["IKRequest", "IKResult", "IKMetrics", "IKSolver"]
