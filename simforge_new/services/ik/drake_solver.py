"""Drake-backed IK solver aligned with the Simforge service contracts."""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple, Sequence

import numpy as np

from .base import IKRequest, IKResult, IKMetrics, IKSolver
from ...core.models import RobotProfile

try:  # pragma: no cover - optional dependency
    from simforge.ik_drake import DrakeIKCache, DrakeIKOptions, solve_ik_drake

    _DRAKE_AVAILABLE = True
except Exception as exc:  # pragma: no cover - IK optional at runtime
    DrakeIKCache = None  # type: ignore
    DrakeIKOptions = None  # type: ignore
    solve_ik_drake = None  # type: ignore
    _DRAKE_AVAILABLE = False
    _DRAKE_IMPORT_ERROR = exc
else:
    _DRAKE_IMPORT_ERROR = None


class DrakeIKSolver(IKSolver):
    """Adapter that wraps the legacy Drake IK pipeline behind the new interface."""

    def __init__(self, profile: RobotProfile, *, logger: Optional[logging.Logger] = None) -> None:
        if not _DRAKE_AVAILABLE:
            raise RuntimeError(f"Drake IK unavailable: {_DRAKE_IMPORT_ERROR}")

        self.profile = profile
        self.logger = logger or logging.getLogger(f"simforge.ik.drake.{profile.name}")
        self._cache = self._build_cache(profile)
        self._last_solution: Optional[np.ndarray] = None

        metadata = profile.metadata or {}
        self._default_pos_tol = self._as_float(metadata.get("legacy_control_ik_pos_tolerance_m"), default=1e-3)
        self._default_rot_tol = self._as_float(metadata.get("legacy_control_ik_rot_tolerance_deg"), default=1.0)

    # ------------------------------------------------------------------
    # IKSolver API
    # ------------------------------------------------------------------
    def solve(self, request: IKRequest) -> IKResult:
        if request.target.pose is None:
            return IKResult(success=False, solution=None, raw={"reason": "no_pose"})

        position = tuple(float(v) for v in request.target.pose.position)
        orientation = tuple(float(v) for v in request.target.pose.orientation)
        q_seed = self._seed_for_request(request)
        is_state_valid = request.is_state_valid or (lambda _: True)

        opts = DrakeIKOptions(
            pos_tolerance_m=request.position_tolerance_m or self._default_pos_tol,
            rot_tolerance_deg=request.orientation_tolerance_deg or self._default_rot_tol,
            timeout_s=max(request.timeout_s, 1e-3),
            allow_position_only_fallback=not request.prefer_cartesian,
        )

        try:
            solution, info = solve_ik_drake(  # type: ignore[misc]
                self._cache,
                q_seed,
                target_pos_base_m=position,
                target_quat_base_wxyz=orientation,
                is_state_valid=is_state_valid,
                opts=opts,
            )
        except Exception as exc:  # pragma: no cover - Drake internal errors
            self.logger.exception("Drake IK invocation failed: %s", exc)
            return IKResult(success=False, solution=None, raw={"reason": "exception", "error": str(exc)})

        if solution is None:
            return IKResult(success=False, solution=None, raw=info)

        self._last_solution = np.asarray(solution, dtype=np.float64)
        metrics = self._compute_metrics(solution, position, orientation, info)
        return IKResult(success=True, solution=tuple(float(v) for v in solution), metrics=metrics, raw=info)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def forward_kinematics(self, joints: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Return end-effector pose (position, quaternion wxyz) for given joints."""
        context = self._cache.plant.CreateDefaultContext()
        total_dofs = self._cache.plant.num_positions()
        q = np.zeros(total_dofs, dtype=np.float64)
        joints_arr = np.asarray(joints, dtype=np.float64)
        count = min(joints_arr.size, total_dofs)
        q[:count] = joints_arr[:count]
        self._cache.plant.SetPositions(context, q)
        transform = self._cache.plant.CalcRelativeTransform(
            context, self._cache.base_frame, self._cache.ee_frame
        )
        pos = transform.translation().copy()
        quat = transform.rotation().ToQuaternion()
        quat_wxyz = np.array(
            [quat.w(), quat.x(), quat.y(), quat.z()],
            dtype=np.float64,
        )
        return pos, quat_wxyz

    def _seed_for_request(self, request: IKRequest) -> np.ndarray:
        if request.seed is not None:
            return np.asarray(request.seed, dtype=np.float64)
        if self._last_solution is not None:
            return self._last_solution.copy()
        if request.robot.initial_joint_positions_deg:
            return np.array([math.radians(float(v)) for v in request.robot.initial_joint_positions_deg], dtype=np.float64)
        dof = self._cache.plant.num_positions()
        return np.zeros(dof, dtype=np.float64)

    def _compute_metrics(
        self,
        solution: Tuple[float, ...],
        target_pos: Tuple[float, float, float],
        target_quat_wxyz: Tuple[float, float, float, float],
        info: dict,
    ) -> IKMetrics:
        try:
            context = self._cache.plant.CreateDefaultContext()
            self._cache.plant.SetPositions(context, solution)
            transform = self._cache.plant.CalcRelativeTransform(
                context, self._cache.base_frame, self._cache.ee_frame
            )
            pos_err = np.linalg.norm(transform.translation() - np.asarray(target_pos, dtype=np.float64))

            quat_current = transform.rotation().ToQuaternion()
            q_current = np.array(
                [quat_current.w(), quat_current.x(), quat_current.y(), quat_current.z()],
                dtype=np.float64,
            )
            q_current /= np.linalg.norm(q_current)
            q_target = np.asarray(target_quat_wxyz, dtype=np.float64)
            norm = np.linalg.norm(q_target)
            if norm > 0.0:
                q_target /= norm
            dot = float(np.clip(np.abs(np.dot(q_current, q_target)), -1.0, 1.0))
            rot_err = 2.0 * math.degrees(math.acos(dot))
        except Exception:  # pragma: no cover - diagnostics only
            pos_err = float("nan")
            rot_err = float("nan")

        attempts = int(info.get("iters", 1)) if isinstance(info, dict) else 1
        diagnostics: Tuple[str, ...] = ()
        if isinstance(info, dict):
            backend = info.get("backend")
            mode = info.get("ori_mode")
            items = [str(v) for v in (backend, mode) if v]
            diagnostics = tuple(items)
        return IKMetrics(
            position_error_m=float(pos_err),
            orientation_error_deg=float(rot_err),
            attempts=attempts,
            slack_used=info.get("ori_mode") == "position_only" if isinstance(info, dict) else False,
            diagnostics=diagnostics,
        )

    def _build_cache(self, profile: RobotProfile) -> object:
        base_link = self._resolve_base_link(profile)
        ee_link = profile.end_effector_link or "tool0"
        self.logger.debug("Initializing Drake IK cache: urdf=%s base=%s ee=%s", profile.urdf, base_link, ee_link)
        return DrakeIKCache(profile.urdf, base_link=base_link, ee_link=ee_link)

    @staticmethod
    def _resolve_base_link(profile: RobotProfile) -> str:
        metadata = profile.metadata or {}
        if "drake_base_link" in metadata:
            return str(metadata["drake_base_link"])
        urdf_lower = profile.urdf.lower()
        if "meca" in urdf_lower:
            return "meca_base_link"
        return "base_link"

    @staticmethod
    def _as_float(value: object, *, default: float) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return float(default)


__all__ = ["DrakeIKSolver"]
