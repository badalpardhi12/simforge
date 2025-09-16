"""Runtime data structures for robot instances."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple
import numpy as np

from .commands import ControlMode


@dataclass
class ActiveTrajectory:
    """Trajectory currently being executed."""
    waypoints: np.ndarray
    times: np.ndarray
    start_time: float

    def sample(self, now: float) -> Dict[str, Any]:
        """Return the interpolated configuration for absolute time ``now``.

        The dictionary contains:
            ``q``: ndarray of joint positions
            ``done``: whether the trajectory has completed
        """
        if now <= self.start_time:
            return {"q": self.waypoints[0], "done": False}
        elapsed = now - self.start_time
        if elapsed >= float(self.times[-1]):
            return {"q": self.waypoints[-1], "done": True}
        idx = int(np.searchsorted(self.times, elapsed, side="right") - 1)
        idx = max(0, min(idx, len(self.times) - 2))
        t0 = float(self.times[idx])
        t1 = float(self.times[idx + 1])
        if t1 <= t0:
            return {"q": self.waypoints[idx + 1], "done": False}
        a = (elapsed - t0) / (t1 - t0)
        q = (1.0 - a) * self.waypoints[idx] + a * self.waypoints[idx + 1]
        return {"q": q, "done": False}


@dataclass
class RobotRuntime:
    """Aggregated runtime state for a robot."""
    name: str
    config: Any
    mode: ControlMode = ControlMode.JOINT
    joint_targets: List[float] = field(default_factory=list)
    entity: Any = None
    collision_checker: Any = None
    pin_model: Any = None
    pin_data: Any = None
    drake_cache: Any = None
    last_safe_q: Optional[np.ndarray] = None
    last_known_q: Optional[np.ndarray] = None
    active_traj: Optional[ActiveTrajectory] = None
    state_valid_cache: Optional[Any] = None
    last_target_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (pos, quat_wxyz) in BASE
    last_planned_q: Optional[np.ndarray] = None
    pending_pose_validation: bool = False

    def has_collision_checker(self) -> bool:
        return bool(self.collision_checker and getattr(self.collision_checker, "available", False))


__all__ = ["RobotRuntime", "ActiveTrajectory"]
