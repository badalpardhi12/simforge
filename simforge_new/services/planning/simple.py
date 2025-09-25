"""Simple joint-space planner used for tests and fallback scenarios."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .base import PlanOutcome, PlanRequest, PlanResult, PlannerMetrics, MotionPlanner
from ...core.trajectories import JointTrajectory


@dataclass(frozen=True)
class LinearPlannerSettings:
    """Configuration for :class:`LinearMotionPlanner`."""

    max_joint_step: float = 0.1
    nominal_speed: float = 1.0


class LinearMotionPlanner(MotionPlanner):
    """Generates linear interpolation between start and goal joint states."""

    def __init__(self, settings: LinearPlannerSettings | None = None) -> None:
        self.settings = settings or LinearPlannerSettings()

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

        if request.is_state_valid and not request.is_state_valid(tuple(float(v) for v in start)):
            return PlanResult(
                outcome=PlanOutcome.FAILURE,
                trajectory=None,
                metrics=None,
                raw={"reason": "invalid_start"},
            )

        diff = goal - start
        span = float(np.max(np.abs(diff))) if diff.size else 0.0
        if span <= 1e-9:
            trajectory = JointTrajectory(
                joint_names=tuple(f"j{idx}" for idx in range(start.size)),
                times_s=(0.0,),
                positions=(tuple(float(v) for v in start),),
            )
            metrics = PlannerMetrics(
                planner_name="linear",
                duration_s=0.0,
                states_validated=1,
                diagnostics={"span": 0.0},
            )
            return PlanResult(outcome=PlanOutcome.SUCCESS, trajectory=trajectory, metrics=metrics, raw={"solver": "linear"})

        steps = max(1, int(math.ceil(span / max(self.settings.max_joint_step, 1e-6))))
        waypoints = [start]

        for step in range(1, steps + 1):
            alpha = step / steps
            waypoint = start + alpha * diff

            if request.is_state_valid:
                state_tuple = tuple(float(v) for v in waypoint)
                if not request.is_state_valid(state_tuple):
                    return PlanResult(
                        outcome=PlanOutcome.COLLISION,
                        trajectory=None,
                        metrics=None,
                        raw={"step": step, "alpha": float(alpha)},
                    )

            waypoints.append(waypoint)

        times = self._compute_times(len(waypoints), span)
        trajectory = JointTrajectory(
            joint_names=tuple(f"j{idx}" for idx in range(start.size)),
            times_s=times,
            positions=tuple(tuple(float(v) for v in wp) for wp in waypoints),
        )
        metrics = PlannerMetrics(
            planner_name="linear",
            duration_s=0.0,
            states_validated=len(waypoints),
            diagnostics={"span": span},
        )
        return PlanResult(outcome=PlanOutcome.SUCCESS, trajectory=trajectory, metrics=metrics, raw={"solver": "linear"})

    def _compute_times(self, count: int, span: float) -> Tuple[float, ...]:
        if count <= 1:
            return (0.0,)
        total_time = span / max(self.settings.nominal_speed, 1e-6)
        dt = total_time / (count - 1)
        return tuple(float(idx * dt) for idx in range(count))


__all__ = ["LinearPlannerSettings", "LinearMotionPlanner"]
