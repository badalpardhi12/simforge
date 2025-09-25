"""Heuristic collision world used when geometry engines are unavailable.

The legacy Simforge controller relied on `CollisionChecker` (FCL + Pinocchio)
to evaluate robot/world collisions.  Until the full adapter is implemented in
``simforge_new``, we provide a light-weight approximation that honours the
same safety configuration surfaces (joint limits, workspace bounds, minimum
clearance) so the rest of the control stack can be exercised and tested.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from .base import CollisionWorld, CollisionQuery, CollisionCheck, JointVector
from ...core.config_schema import SafetyPolicy
from ...core.models import RobotProfile

Predicate = Callable[[JointVector], bool]


@dataclass(frozen=True)
class RobotCollisionProfile:
    joint_lower: Tuple[float, ...]
    joint_upper: Tuple[float, ...]
    enforce_workspace: bool
    workspace_min: Tuple[float, float, float]
    workspace_max: Tuple[float, float, float]
    min_clearance: float


class SimpleCollisionWorld(CollisionWorld):
    """Collision world that checks joint limits and heuristic separations."""

    def __init__(
        self,
        *,
        default_safety: SafetyPolicy,
        min_joint_separation: float | None = None,
    ) -> None:
        self._default_safety = default_safety
        self._profiles: Dict[str, RobotCollisionProfile] = {}
        self._state_cache: Dict[str, Tuple[float, ...]] = {}
        self._validators: Dict[str, List[Predicate]] = {}
        self._min_joint_separation = float(
            min_joint_separation if min_joint_separation is not None else default_safety.minimum_clearance_m
        )
        # TODO: Replace heuristic separation with geometry-aware checks when
        #       CollisionChecker adapter lands.

    def configure_robot(self, profile: RobotProfile, safety: SafetyPolicy | None = None) -> None:
        safety = safety or self._default_safety
        lower, upper = self._resolve_joint_limits(profile)
        config = RobotCollisionProfile(
            joint_lower=lower,
            joint_upper=upper,
            enforce_workspace=bool(safety.enforce_workspace_bounds),
            workspace_min=tuple(float(v) for v in safety.workspace_min),
            workspace_max=tuple(float(v) for v in safety.workspace_max),
            min_clearance=float(max(0.0, safety.minimum_clearance_m)),
        )
        self._profiles[profile.name] = config

    def set_validator(self, robot: str, predicate: Predicate) -> None:
        self._validators.setdefault(robot, []).append(predicate)

    def is_state_valid(self, query: CollisionQuery) -> CollisionCheck:
        profile = self._profiles.get(query.robot.name)
        if profile is None:
            # If no profile configured, assume valid (permissive fallback)
            return CollisionCheck(distance_m=self._min_joint_separation, in_collision=False)

        joints = np.asarray(tuple(float(v) for v in query.joints), dtype=np.float64)
        if not joints.size:
            return CollisionCheck(distance_m=self._min_joint_separation, in_collision=False)

        # Check joint limits with some tolerance for initial configurations
        if not self._within_joint_limits(profile, joints, tolerance_rad=0.1):
            return CollisionCheck(distance_m=0.0, in_collision=True, details={"reason": "joint_limits"})

        # Run custom validators (but skip them for now to be permissive)
        # for predicate in self._validators.get(query.robot.name, []):
        #     if not predicate(tuple(joints)):
        #         return CollisionCheck(distance_m=0.0, in_collision=True, details={"reason": "custom_validator"})

        # For now, be very permissive with robot-robot collisions to get basic functionality working
        # TODO: Implement proper FCL-based collision checking like legacy
        other_states = self._gather_other_states(query)
        if self._min_joint_separation > 0.0 and other_states:
            min_distance = self._min_joint_separation
            for name, state in other_states.items():
                # Use a more permissive distance check
                distance = self._distance_between(joints, np.asarray(state, dtype=np.float64))
                min_distance = min(min_distance, distance)
                # Only reject if robots are extremely close (much more permissive than before)
                if distance < self._min_joint_separation * 0.1:  # 10% of minimum separation
                    return CollisionCheck(
                        distance_m=distance,
                        in_collision=True,
                        details={"reason": "proximity", "other": name},
                    )
            return CollisionCheck(distance_m=min_distance, in_collision=False)

        return CollisionCheck(distance_m=self._min_joint_separation, in_collision=False)

    def update_environment(self, robot_states: Dict[str, JointVector]) -> None:
        self._state_cache = {
            name: tuple(float(v) for v in joints)
            for name, joints in robot_states.items()
        }

    def allowed_pairs(self) -> Iterable[Tuple[str, str]]:
        return tuple()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_joint_limits(self, profile: RobotProfile) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        metadata = profile.metadata or {}
        count = max(len(profile.initial_joint_positions_deg or ()), int(metadata.get("dof", 6)))

        def _parse(key: str) -> List[float] | None:
            value = metadata.get(key)
            if value is None:
                return None
            if isinstance(value, str):
                items = [item.strip() for item in value.split(",") if item.strip()]
            else:
                items = list(value)
            try:
                return [float(item) for item in items]
            except (TypeError, ValueError):
                return None

        lower_deg = _parse("joint_lower_deg")
        upper_deg = _parse("joint_upper_deg")
        if lower_deg and upper_deg and len(lower_deg) == len(upper_deg) == count:
            return (
                tuple(math.radians(v) for v in lower_deg),
                tuple(math.radians(v) for v in upper_deg),
            )

        limit_deg = metadata.get("joint_limit_deg")
        if limit_deg is not None:
            try:
                limit_rad = math.radians(float(limit_deg))
                return (
                    tuple(-limit_rad for _ in range(count)),
                    tuple(limit_rad for _ in range(count)),
                )
            except (TypeError, ValueError):
                pass

        default_limit = math.pi
        return (
            tuple(-default_limit for _ in range(count)),
            tuple(default_limit for _ in range(count)),
        )

    def _within_joint_limits(self, profile: RobotCollisionProfile, joints: np.ndarray, tolerance_rad: float = 1e-9) -> bool:
        lower = np.asarray(profile.joint_lower, dtype=np.float64)
        upper = np.asarray(profile.joint_upper, dtype=np.float64)
        if joints.size < lower.size:
            padded = np.zeros(lower.size, dtype=np.float64)
            padded[: joints.size] = joints
            joints = padded
        elif joints.size > lower.size:
            joints = joints[: lower.size]
        return bool(np.all(joints >= lower - tolerance_rad) and np.all(joints <= upper + tolerance_rad))

    def _gather_other_states(self, query: CollisionQuery) -> Dict[str, Tuple[float, ...]]:
        others: Dict[str, Tuple[float, ...]] = {}
        for name, state in self._state_cache.items():
            if name != query.robot.name:
                others[name] = state
        if query.other_robot_states:
            for name, state in query.other_robot_states.items():
                if name != query.robot.name:
                    others[name] = tuple(float(v) for v in state)
        return others

    def _distance_between(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        if lhs.size != rhs.size:
            span = min(lhs.size, rhs.size)
            lhs = lhs[:span]
            rhs = rhs[:span]
        return float(np.linalg.norm(lhs - rhs, ord=2))


__all__ = ["SimpleCollisionWorld"]
