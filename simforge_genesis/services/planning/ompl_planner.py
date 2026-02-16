"""OMPL-backed motion planner for the new Simforge architecture."""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from .base import MotionPlanner, PlanOutcome, PlanRequest, PlanResult, PlannerMetrics
from ...core.models import RobotProfile
from ...core.trajectories import JointTrajectory
from ..collision.base import CollisionWorld

try:  # pragma: no cover - OMPL optional at runtime
    from simforge.path_planner import (
        default_joint_planner_specs,
        ompl_parallel_plans,
        plan_joint_path,
    )

    _OMPL_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    default_joint_planner_specs = None  # type: ignore
    ompl_parallel_plans = None  # type: ignore
    plan_joint_path = None  # type: ignore
    _OMPL_AVAILABLE = False
    _OMPL_IMPORT_ERROR = exc
else:
    _OMPL_IMPORT_ERROR = None


def _plan_linear(start: np.ndarray, goal: np.ndarray, *, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    if plan_joint_path is not None:
        return plan_joint_path(start, goal, resolution=resolution)  # type: ignore[misc]
    waypoints = np.linspace(start, goal, resolution)
    times = np.linspace(0.0, 1.0, resolution)
    return waypoints, times


class OMPLMotionPlanner(MotionPlanner):
    """Adapter that reuses the legacy OMPL planning stack."""

    def __init__(
        self,
        robot: RobotProfile,
        *,
        collision_world: Optional[CollisionWorld],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.robot = robot
        self.collision_world = collision_world
        self.logger = logger or logging.getLogger(f"simforge.planning.ompl.{robot.name}")

        self._lower_limits, self._upper_limits = self._derive_joint_limits()
        self._planner_specs = self._build_planner_specs()

    # ------------------------------------------------------------------
    # MotionPlanner API
    # ------------------------------------------------------------------
    def plan(self, request: PlanRequest) -> PlanResult:
        start = np.asarray(request.start, dtype=np.float64)
        goal = np.asarray(request.goal, dtype=np.float64)

        if start.shape != goal.shape:
            return PlanResult(
                outcome=PlanOutcome.INVALID_GOAL,
                trajectory=None,
                metrics=None,
                raw={"reason": "shape_mismatch"},
            )

        validator = request.is_state_valid or (lambda _: True)

        # Quickly check start/goal validity before heavy planning effort.
        if not validator(tuple(float(v) for v in start)):
            return PlanResult(
                outcome=PlanOutcome.FAILURE,
                trajectory=None,
                metrics=None,
                raw={"reason": "invalid_start"},
            )
        if not validator(tuple(float(v) for v in goal)):
            return PlanResult(
                outcome=PlanOutcome.INVALID_GOAL,
                trajectory=None,
                metrics=None,
                raw={"reason": "goal_in_collision"},
            )

        best = None
        if _OMPL_AVAILABLE and self._planner_specs:
            try:
                plans = ompl_parallel_plans(  # type: ignore[misc]
                    self._planner_specs,
                    start,
                    goal,
                    self._lower_limits,
                    self._upper_limits,
                    lambda q: validator(tuple(float(v) for v in q)),
                    timeout_s=max(request.timeout_s, 0.1),
                )
            except Exception as exc:  # pragma: no cover - OMPL failure
                self.logger.warning("OMPL planning failed: %s", exc)
                plans = []
            if plans:
                best = plans[0]

        if best is None:
            # Controlled linear fallback mirroring the legacy controller.
            waypoints, _ = _plan_linear(start, goal, resolution=max(10, start.size * 10))
            if not self._validate_path(waypoints, validator):
                return PlanResult(
                    outcome=PlanOutcome.COLLISION,
                    trajectory=None,
                    metrics=None,
                    raw={"reason": "linear_path_collision"},
                )
            times = self._even_times(waypoints.shape[0], request.timeout_s)
            trajectory = self._to_trajectory(waypoints, times)
            metrics = PlannerMetrics(
                planner_name="linear",
                duration_s=float(times[-1]) if times else 0.0,
                states_validated=waypoints.shape[0],
                clearance_min_m=None,
                diagnostics={"fallback": 1.0},
            )
            return PlanResult(PlanOutcome.SUCCESS, trajectory=trajectory, metrics=metrics, raw={"solver": "linear"})

        planner_name, waypoints, times, cost = best
        if not self._validate_path(waypoints, validator):
            return PlanResult(
                outcome=PlanOutcome.COLLISION,
                trajectory=None,
                metrics=None,
                raw={"reason": "ompl_path_collision", "planner": planner_name},
            )

        trajectory = self._to_trajectory(waypoints, times)
        metrics = PlannerMetrics(
            planner_name=planner_name,
            duration_s=trajectory.duration,
            states_validated=waypoints.shape[0],
            clearance_min_m=None,
            diagnostics={"cost": float(cost)},
        )
        return PlanResult(PlanOutcome.SUCCESS, trajectory=trajectory, metrics=metrics, raw={"solver": "ompl"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _derive_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.collision_world is not None and hasattr(self.collision_world, "joint_limits"):
            try:
                limits = self.collision_world.joint_limits(self.robot.name)  # type: ignore[attr-defined]
                if limits is not None:
                    lower, upper = limits
                    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
            except Exception:  # pragma: no cover - defensive
                self.logger.debug("Failed to retrieve joint limits from collision world", exc_info=True)

        dof = self.robot.joint_count
        lower = np.full(dof, -math.pi, dtype=np.float64)
        upper = np.full(dof, math.pi, dtype=np.float64)
        return lower, upper

    def _build_planner_specs(self):
        if not _OMPL_AVAILABLE:
            self.logger.debug("OMPL not available: %s", _OMPL_IMPORT_ERROR)
            return []
        span = float(np.max(self._upper_limits - self._lower_limits))
        range_hint = max(0.1, span * 0.25)
        try:
            return default_joint_planner_specs(range_hint)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - OMPL failure
            self.logger.warning("Failed to construct planner specs: %s", exc)
            return []

    def _validate_path(self, waypoints: np.ndarray, validator) -> bool:
        for waypoint in waypoints:
            if not validator(tuple(float(v) for v in waypoint)):
                return False
        return True

    def _even_times(self, count: int, total_hint: float) -> Tuple[float, ...]:
        if count <= 1:
            return (0.0,)
        total = max(total_hint, 2.0)
        dt = total / max(count - 1, 1)
        return tuple(idx * dt for idx in range(count))

    def _to_trajectory(self, waypoints: np.ndarray, times: np.ndarray | Tuple[float, ...]) -> JointTrajectory:
        if isinstance(times, np.ndarray):
            times_seq = tuple(float(v) for v in times)
        else:
            times_seq = tuple(float(v) for v in times)
        positions = tuple(tuple(float(v) for v in row) for row in waypoints)
        joint_names = tuple(f"j{idx}" for idx in range(waypoints.shape[1]))
        return JointTrajectory(joint_names, times_seq, positions)


__all__ = ["OMPLMotionPlanner"]
