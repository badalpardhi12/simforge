"""Collision world backed by FCL and Pinocchio."""
from __future__ import annotations

import logging
from dataclasses import dataclass
import threading
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .base import CollisionCheck, CollisionQuery, CollisionWorld
from .simple import SimpleCollisionWorld
from ...core.config_schema import SafetyPolicy, WorldObjectType
from ...core.models import EnvironmentSpec, RobotProfile

try:  # Optional dependencies resolved lazily at runtime
    from simforge.collision_checker import CollisionChecker as FCLChecker
    _FCL_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _FCL_AVAILABLE = False

try:
    import pinocchio as pin  # type: ignore
    _PIN_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    pin = None  # type: ignore
    _PIN_AVAILABLE = False


def _normalize_quaternion(quat: Tuple[float, float, float, float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def _quat_to_rpy(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    qw, qx, qy, qz = _normalize_quaternion(quat)
    t0 = 2.0 * (qw * qx + qy * qz)
    t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (qw * qy - qz * qx)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(t3, t4)
    return float(roll), float(pitch), float(yaw)


@dataclass(frozen=True)
class _RobotKinematics:
    checker: Optional[FCLChecker]
    model: Optional[object]
    urdf: str
    base_position: Tuple[float, float, float]
    base_rpy: Tuple[float, float, float]


class FCLCollisionWorld(CollisionWorld):
    """Geometry-aware collision world used by the Simforge control stack."""

    def __init__(
        self,
        profiles: Dict[str, RobotProfile],
        *,
        spec: EnvironmentSpec,
        safety: SafetyPolicy,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger("simforge.collision.fcl")
        self._profiles = dict(profiles)
        self._spec = spec
        self._safety = safety
        self._min_clearance = float(max(0.0, safety.minimum_clearance_m))
        self._latest_states: Dict[str, Tuple[float, ...]] = {}
        self._world_boxes, self._world_meshes = self._build_world_geometry()

        self._fallback = SimpleCollisionWorld(
            default_safety=safety,
            min_joint_separation=self._min_clearance,
        )
        for profile in profiles.values():
            self._fallback.configure_robot(profile, safety)

        self._robots: Dict[str, _RobotKinematics] = {}
        self._lock = threading.RLock()
        self._enable_env_robot_updates = False
        self._env_update_pairs_logged: set[Tuple[str, str]] = set()
        self._latest_states: Dict[str, Tuple[float, ...]] = {}
        if not _FCL_AVAILABLE or not _PIN_AVAILABLE:
            missing = []
            if not _FCL_AVAILABLE:
                missing.append("FCL/trimesh")
            if not _PIN_AVAILABLE:
                missing.append("Pinocchio")
            self._logger.warning(
                "Geometry collision checks unavailable (%s); using heuristic fallback.",
                ", ".join(missing) or "unknown",
            )
            return

        for name, profile in profiles.items():
            kin = self._initialize_robot(profile)
            if kin is None:
                continue
            self._robots[name] = kin
            initial_state = self._initial_joint_state(profile, kin)
            if initial_state is not None:
                self._latest_states[name] = initial_state

        self._enable_env_robot_updates = (
            len(self._robots) > 1 and _FCL_AVAILABLE and _PIN_AVAILABLE
        )
        if self._enable_env_robot_updates:
            for name, kin in self._robots.items():
                checker = kin.checker
                if checker is None or not checker.available:
                    continue
                for other_name, other_kin in self._robots.items():
                    if other_name == name or other_kin.checker is None:
                        continue
                    try:
                        checker.register_env_robot(other_name, other_kin.urdf)
                    except Exception as exc:  # pragma: no cover
                        self._logger.debug(
                            "Failed to register %s inside %s checker: %s",
                            other_name,
                            name,
                            exc,
                        )
        elif len(self._profiles) > 1:
            self._logger.warning(
                "Multi-robot FCL environment updates unavailable (missing dependencies or robot initialisation failed); "
                "falling back to heuristic avoidance for inter-robot collisions."
            )

        if self._latest_states:
            try:
                self._fallback.update_environment(self._latest_states)
            except Exception:  # pragma: no cover - defensive
                pass
            self._apply_env_robot_states_locked()

    # ------------------------------------------------------------------
    # CollisionWorld API
    # ------------------------------------------------------------------
    def is_state_valid(self, query: CollisionQuery) -> CollisionCheck:
        with self._lock:
            return self._is_state_valid_locked(query)

    def _is_state_valid_locked(self, query: CollisionQuery) -> CollisionCheck:
        if query.robot.name not in self._robots:
            return self._fallback.is_state_valid(query)

        kin = self._robots[query.robot.name]
        checker = kin.checker
        model = kin.model
        if checker is None or model is None or not checker.available:
            return self._fallback.is_state_valid(query)

        q = self._coerce_state(query.joints, model.nq)
        try:
            in_collision = checker.in_collision_from_pin(model, model.createData(), q)
        except Exception as exc:  # pragma: no cover
            self._logger.warning("Collision check failed for %s: %s", query.robot.name, exc)
            return self._fallback.is_state_valid(query)

        details = {"source": "fcl"}
        info_fn = getattr(checker, "get_last_collision_info", None)
        if callable(info_fn):
            try:
                info = info_fn()
            except Exception:  # pragma: no cover - defensive
                info = None
            if info:
                details["collision"] = info

        if in_collision:
            return CollisionCheck(distance_m=0.0, in_collision=True, details=details)
        return CollisionCheck(distance_m=self._min_clearance, in_collision=False, details=details)

    def update_environment(self, robot_states: Dict[str, Tuple[float, ...]]) -> None:
        with self._lock:
            self._latest_states = {
                name: tuple(float(v) for v in values)
                for name, values in (robot_states or {}).items()
            }
            self._fallback.update_environment(robot_states)
            self._apply_env_robot_states_locked()

    def allowed_pairs(self) -> Iterable[Tuple[str, str]]:
        return self._fallback.allowed_pairs()

    def joint_limits(self, robot: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            kin = self._robots.get(robot)
        if kin is None or kin.model is None:
            return None
        model = kin.model
        try:
            lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
            upper = np.asarray(model.upperPositionLimit, dtype=np.float64)
        except Exception:  # pragma: no cover - defensive
            return None
        if lower.size == 0 or upper.size == 0:
            return None
        return lower, upper

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _initialize_robot(self, profile: RobotProfile) -> Optional[_RobotKinematics]:
        urdf_path = str(profile.urdf)
        base_position = tuple(float(v) for v in profile.mount.position)
        base_rpy = _quat_to_rpy(profile.mount.orientation)
        allowed = [] if profile.control.allow_self_collision else None

        try:
            checker = FCLChecker(
                urdf_path,
                self._logger,
                base_position=base_position,
                base_orientation_rpy=base_rpy,
                allowed_link_pairs=allowed,
                world_allowed_pairs=[],
                world_boxes=self._world_boxes,
                world_meshes=self._world_meshes,
                ground_plane_z=0.0,
                collision_mesh_shrink=1.0,
            )
        except Exception as exc:  # pragma: no cover
            self._logger.warning("Collision checker init failed for %s: %s", profile.name, exc)
            checker = None

        model = None
        if checker is not None and checker.available and _PIN_AVAILABLE:
            try:
                model = pin.buildModelFromUrdf(urdf_path)
            except Exception as exc:  # pragma: no cover
                self._logger.warning("Pinocchio model failed for %s: %s", profile.name, exc)
                model = None

        return _RobotKinematics(
            checker=checker,
            model=model,
            urdf=urdf_path,
            base_position=base_position,
            base_rpy=base_rpy,
        )

    def _initial_joint_state(
        self,
        profile: RobotProfile,
        kin: Optional[_RobotKinematics],
    ) -> Optional[Tuple[float, ...]]:
        joints_deg = profile.initial_joint_positions_deg
        if joints_deg:
            values = tuple(float(np.deg2rad(v)) for v in joints_deg)  # type: ignore[name-defined]
        else:
            dof = profile.joint_count if profile.joint_count > 0 else 0
            if dof == 0:
                return None
            values = tuple(0.0 for _ in range(dof))

        if kin and kin.model is not None:
            nq = int(getattr(kin.model, "nq", len(values)))
            if nq and len(values) != nq:
                padded = [0.0] * nq
                span = min(nq, len(values))
                padded[:span] = values[:span]
                values = tuple(padded)
        return values

    def _coerce_state(self, joints: Tuple[float, ...], dimension: int) -> np.ndarray:
        arr = np.asarray(joints, dtype=np.float64).flatten()
        if arr.size == dimension:
            return arr
        padded = np.zeros(dimension, dtype=np.float64)
        span = min(dimension, arr.size)
        if span:
            padded[:span] = arr[:span]
        return padded

    def _apply_env_robot_states_locked(self) -> None:
        if not self._enable_env_robot_updates:
            return

        for subject_name, subject_kin in self._robots.items():
            checker = subject_kin.checker
            if checker is None or not checker.available:
                continue

            for other_name, state in self._latest_states.items():
                if other_name == subject_name:
                    continue
                other = self._robots.get(other_name)
                if other is None or other.model is None:
                    continue

                if (subject_name, other_name) not in self._env_update_pairs_logged:
                    self._logger.info(
                        "FCL env update pair ready: %s <- %s (state=%d, nq=%d)",
                        subject_name,
                        other_name,
                        len(state),
                        other.model.nq if hasattr(other.model, "nq") else -1,
                    )
                    self._env_update_pairs_logged.add((subject_name, other_name))

                q_other = self._coerce_state(state, other.model.nq)
                try:
                    checker.update_env_robot_from_pin(
                        other_name,
                        other.model,
                        other.model.createData(),
                        q_other,
                        base_position=other.base_position,
                        base_orientation_rpy=other.base_rpy,
                    )
                except Exception as exc:  # pragma: no cover
                    self._logger.warning(
                        "Failed to update %s inside %s checker; disabling env updates: %s",
                        other_name,
                        subject_name,
                        exc,
                    )
                    self._enable_env_robot_updates = False
                    return

    def _build_world_geometry(self) -> Tuple[list, list]:
        boxes = []
        meshes = []
        for obj in self._spec.world.objects:
            if not obj.collision_enabled:
                continue
            name = obj.name or "object"
            position = tuple(float(v) for v in obj.pose_position)
            orientation = tuple(float(v) for v in obj.pose_orientation_rpy)

            if obj.type == WorldObjectType.BOX:
                tag = f"box:{name}" if name else "box"
                size = obj.size or (0.1, 0.1, 0.1)
                boxes.append((tag, tuple(float(v) for v in size), position, orientation))
            elif obj.type == WorldObjectType.SPHERE:
                tag = f"sphere:{name}" if name else "sphere"
                radius = float(obj.radius or 0.05)
                boxes.append((tag, {"type": "sphere", "radius": radius}, position, orientation))
            elif obj.type == WorldObjectType.PLANE:
                tag = f"plane:{name}" if name else "plane"
                size = obj.size or (10.0, 10.0, 0.02)
                center = list(position)
                center[2] -= float(size[2]) * 0.5
                boxes.append((tag, tuple(float(v) for v in size), tuple(center), orientation))
            elif obj.type == WorldObjectType.URDF and obj.urdf:
                tag = f"urdf:{name}" if name else "urdf"
                meshes.append((tag, str(obj.urdf), position, orientation))
        return boxes, meshes


__all__ = ["FCLCollisionWorld"]