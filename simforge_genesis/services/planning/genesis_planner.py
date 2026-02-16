"""Motion planner built on Genesis path planning APIs."""
from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Tuple
import threading

import numpy as np
import torch

import genesis as gs

from .base import MotionPlanner, PlanOutcome, PlanRequest, PlanResult, PlannerMetrics
from ..collision.base import CollisionWorld
from ..util.joint_limits import resolve_joint_limits, wrap_vector_to_limits
from ...core.models import RobotProfile
from ...core.trajectories import JointTrajectory


def _as_float(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


class GenesisMotionPlanner(MotionPlanner):
    """Joint-space motion planning powered by Genesis RRT variants."""

    def __init__(
        self,
        *,
        profile: RobotProfile,
        entity,
        collision_world: Optional[CollisionWorld],
        logger: Optional[logging.Logger] = None,
        lock: Optional[threading.RLock] = None,
    ) -> None:
        self.profile = profile
        self.entity = entity
        self.collision_world = collision_world
        self.logger = logger or logging.getLogger(f"simforge.planning.genesis.{profile.name}")
        self._lock = lock or threading.RLock()

        metadata = profile.metadata or {}
        self._planner_name = str(metadata.get("genesis_planner", "RRTConnect"))
        self._resolution = _as_float(metadata.get("genesis_plan_resolution"), 0.05)
        self._smooth_path = _as_bool(metadata.get("genesis_plan_smooth", True), True)

        self._waypoints_adaptive = _as_bool(metadata.get("genesis_plan_waypoints_adaptive", True), True)
        default_waypoints = int(_as_float(metadata.get("genesis_plan_waypoints"), 300))
        self._waypoint_spacing = max(1e-4, _as_float(metadata.get("genesis_plan_waypoint_spacing"), 0.03))
        self._num_waypoints_min = max(2, int(_as_float(metadata.get("genesis_plan_waypoints_min"), 6)))
        self._num_waypoints_max = max(self._num_waypoints_min, int(_as_float(metadata.get("genesis_plan_waypoints_max"), default_waypoints)))
        # Minimum delta used when pruning near-identical waypoints from the backend result
        self._min_waypoint_delta = max(1e-6, _as_float(metadata.get("genesis_plan_min_waypoint_delta"), 5e-4))
        self._num_waypoints_default = int(max(self._num_waypoints_min, min(self._num_waypoints_max, default_waypoints)))

        joint_lower, joint_upper = resolve_joint_limits(profile, entity)
        self._joint_lower = joint_lower
        self._joint_upper = joint_upper

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

        start_tensor = self._tensor(start)
        goal_tensor = self._tensor(goal)

        timeout = request.timeout_s if request.timeout_s is not None and request.timeout_s > 0 else None
        num_waypoints = self._select_waypoint_count(start, goal)

        try:
            with self._lock:
                path, valid_mask = self.entity.plan_path(
                    goal_tensor,
                    qpos_start=start_tensor,
                    resolution=self._resolution,
                    timeout=timeout,
                    smooth_path=self._smooth_path,
                    num_waypoints=num_waypoints,
                    planner=self._planner_name,
                    return_valid_mask=True,
                    ignore_collision=False,
                )
        except Exception as exc:  # pragma: no cover - Genesis internal errors
            self.logger.exception("Genesis planning failed for %s: %s", self.profile.name, exc)
            return PlanResult(PlanOutcome.FAILURE, trajectory=None, metrics=None, raw={"reason": "exception", "error": str(exc)})

        valid = bool(valid_mask if isinstance(valid_mask, bool) else valid_mask.item())
        if not valid:
            return PlanResult(
                outcome=PlanOutcome.COLLISION,
                trajectory=None,
                metrics=None,
                raw={"reason": "planning_invalid"},
            )

        waypoints = path.detach().cpu().numpy()
        if waypoints.ndim == 3:
            waypoints = waypoints[:, 0, :]

        waypoints = self._unwrap_waypoints(waypoints)
        waypoints = self._enforce_joint_limits(waypoints)
        waypoints = self._deduplicate_waypoints(waypoints)

        if not self._validate_waypoints(waypoints, validator):
            return PlanResult(
                outcome=PlanOutcome.COLLISION,
                trajectory=None,
                metrics=None,
                raw={"reason": "validator_reject"},
            )

        times = self._even_times(waypoints.shape[0], request.timeout_s)
        trajectory = self._to_trajectory(waypoints, times)
        metrics = PlannerMetrics(
            planner_name=self._planner_name,
            duration_s=trajectory.duration,
            states_validated=waypoints.shape[0],
            clearance_min_m=None,
            diagnostics={"backend": 1.0},
        )

        return PlanResult(PlanOutcome.SUCCESS, trajectory=trajectory, metrics=metrics, raw={"solver": "genesis"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tensor(self, values: Sequence[float]) -> torch.Tensor:
        arr = np.asarray(tuple(float(v) for v in values), dtype=np.float64)
        dof = int(getattr(self.entity, "n_qs", arr.size))
        if arr.size < dof:
            arr = np.pad(arr, (0, dof - arr.size), constant_values=0.0)
        elif arr.size > dof:
            arr = arr[:dof]
        return torch.as_tensor(arr, dtype=gs.tc_float, device=gs.device)

    def _validate_waypoints(self, waypoints: np.ndarray, validator) -> bool:
        for qp in waypoints:
            if not validator(tuple(float(v) for v in qp)):
                return False
        return True

    def _select_waypoint_count(self, start: np.ndarray, goal: np.ndarray) -> int:
        if not self._waypoints_adaptive:
            return self._num_waypoints_default

        start_arr = np.asarray(start, dtype=np.float64)
        goal_arr = np.asarray(goal, dtype=np.float64)
        total_motion = float(np.linalg.norm(goal_arr - start_arr, ord=1))

        if not math.isfinite(total_motion):
            return self._num_waypoints_default

        count = int(math.ceil(total_motion / self._waypoint_spacing)) + 1
        if count <= 2:
            return 2
        count = max(self._num_waypoints_min, count)
        count = min(self._num_waypoints_max, count)
        return max(2, count)

    def _deduplicate_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        if waypoints.ndim != 2 or waypoints.shape[0] <= 1:
            return waypoints

        filtered = [waypoints[0]]
        for idx in range(1, waypoints.shape[0]):
            current = waypoints[idx]
            if np.linalg.norm(current - filtered[-1]) < self._min_waypoint_delta:
                continue
            filtered.append(current)

        if np.linalg.norm(waypoints[-1] - filtered[-1]) >= self._min_waypoint_delta:
            filtered.append(waypoints[-1])
        else:
            filtered[-1] = waypoints[-1]

        if len(filtered) < 2:
            filtered.append(waypoints[-1])

        return np.asarray(filtered, dtype=np.float64)

    def _unwrap_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        if waypoints.ndim != 2 or waypoints.shape[0] <= 1:
            return waypoints

        unwrapped = waypoints.copy()
        period = 2.0 * math.pi
        prev = unwrapped[0].copy()
        for idx in range(1, unwrapped.shape[0]):
            current = unwrapped[idx]
            delta = current - prev
            shifts = np.round(delta / period)
            current -= shifts * period
            residual = current - prev
            current -= np.where(residual > math.pi, period, 0.0)
            current += np.where(residual < -math.pi, period, 0.0)
            unwrapped[idx] = current
            prev = current
        return unwrapped

    def _enforce_joint_limits(self, waypoints: np.ndarray) -> np.ndarray:
        if (
            waypoints.ndim != 2
            or waypoints.shape[0] == 0
            or self._joint_lower is None
            or self._joint_upper is None
        ):
            return waypoints

        result = waypoints.copy()
        for idx in range(result.shape[0]):
            reference = result[idx - 1] if idx > 0 else result[idx]
            result[idx] = wrap_vector_to_limits(result[idx], reference, self._joint_lower, self._joint_upper)
        return result

    def _even_times(self, count: int, total_hint: Optional[float]) -> Tuple[float, ...]:
        if count <= 1:
            return (0.0,)
        total = max(total_hint or 2.0, 2.0)
        dt = total / max(count - 1, 1)
        return tuple(idx * dt for idx in range(count))

    def _to_trajectory(self, waypoints: np.ndarray, times: Sequence[float]) -> JointTrajectory:
        positions = tuple(tuple(float(v) for v in row) for row in waypoints)
        joint_names = tuple(f"j{idx}" for idx in range(waypoints.shape[1]))
        times_seq = tuple(float(v) for v in times)
        return JointTrajectory(joint_names, times_seq, positions)


__all__ = ["GenesisMotionPlanner"]
