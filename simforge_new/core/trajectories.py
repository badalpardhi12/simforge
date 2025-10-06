"""Trajectory utilities shared by planning and execution layers."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TrajectoryPoint:
    """A single sampled point on a trajectory."""

    positions: Tuple[float, ...]
    velocities: Optional[Tuple[float, ...]] = None
    accelerations: Optional[Tuple[float, ...]] = None
    time: float = 0.0


@dataclass(frozen=True)
class TrajectorySegment:
    """A simple segment between two trajectory points."""

    start: TrajectoryPoint
    end: TrajectoryPoint

    def interpolate(self, alpha: float) -> TrajectoryPoint:
        alpha_clamped = float(max(0.0, min(1.0, alpha)))
        positions = tuple(
            (1.0 - alpha_clamped) * s + alpha_clamped * e
            for s, e in zip(self.start.positions, self.end.positions)
        )
        if self.start.velocities is not None and self.end.velocities is not None:
            velocities: Optional[Tuple[float, ...]] = tuple(
                (1.0 - alpha_clamped) * s + alpha_clamped * e
                for s, e in zip(self.start.velocities, self.end.velocities)
            )
        else:
            velocities = None
        if self.start.accelerations is not None and self.end.accelerations is not None:
            accelerations: Optional[Tuple[float, ...]] = tuple(
                (1.0 - alpha_clamped) * s + alpha_clamped * e
                for s, e in zip(self.start.accelerations, self.end.accelerations)
            )
        else:
            accelerations = None
        time = (1.0 - alpha_clamped) * self.start.time + alpha_clamped * self.end.time
        return TrajectoryPoint(positions=positions, velocities=velocities, accelerations=accelerations, time=time)


@dataclass(frozen=True)
class JointTrajectory:
    """Trajectory defined in joint space."""

    joint_names: Tuple[str, ...]
    times_s: Tuple[float, ...]
    positions: Tuple[Tuple[float, ...], ...]
    velocities: Optional[Tuple[Tuple[float, ...], ...]] = None
    accelerations: Optional[Tuple[Tuple[float, ...], ...]] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = len(self.times_s)
        if count != len(self.positions):
            raise ValueError("times_s and positions length mismatch")
        if count == 0:
            raise ValueError("trajectory needs at least one waypoint")
        if any(t_next < t_curr for t_curr, t_next in zip(self.times_s, self.times_s[1:])):
            raise ValueError("times_s must be non-decreasing")
        joint_dim = len(self.joint_names)
        if joint_dim == 0:
            raise ValueError("joint_names must not be empty")
        for position in self.positions:
            if len(position) != joint_dim:
                raise ValueError("each position must match number of joint names")
        if self.velocities is not None and len(self.velocities) != count:
            raise ValueError("velocities length mismatch")
        if self.accelerations is not None and len(self.accelerations) != count:
            raise ValueError("accelerations length mismatch")

    @property
    def point_count(self) -> int:
        return len(self.times_s)

    @property
    def duration(self) -> float:
        return float(self.times_s[-1]) if self.times_s else 0.0

    def sample(self, t: float) -> np.ndarray:
        times = self.times_s
        if not times:
            raise ValueError("Empty trajectory")
        if t <= times[0]:
            return np.asarray(self.positions[0], dtype=np.float64)
        if t >= times[-1]:
            return np.asarray(self.positions[-1], dtype=np.float64)

        idx = bisect_left(times, t)
        idx = min(max(idx, 1), len(times) - 1)
        t0, t1 = times[idx - 1], times[idx]
        span = t1 - t0
        alpha = 0.0 if span <= 1e-9 else (t - t0) / span
        p0 = np.asarray(self.positions[idx - 1], dtype=np.float64)
        p1 = np.asarray(self.positions[idx], dtype=np.float64)
        return (1.0 - alpha) * p0 + alpha * p1

    def as_points(self) -> Tuple[TrajectoryPoint, ...]:
        points: list[TrajectoryPoint] = []
        for i, time in enumerate(self.times_s):
            vel = self.velocities[i] if self.velocities is not None else None
            acc = self.accelerations[i] if self.accelerations is not None else None
            points.append(
                TrajectoryPoint(
                    positions=tuple(float(v) for v in self.positions[i]),
                    velocities=tuple(float(v) for v in vel) if vel is not None else None,
                    accelerations=tuple(float(v) for v in acc) if acc is not None else None,
                    time=float(time),
                )
            )
        return tuple(points)


@dataclass(frozen=True)
class CartesianTrajectory:
    """Trajectory defined in Cartesian space."""

    positions: Tuple[Tuple[float, ...], ...]
    orientations: Tuple[Tuple[float, ...], ...]
    times: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.positions) != len(self.times):
            raise ValueError("positions and times length mismatch")
        if len(self.orientations) != len(self.times):
            raise ValueError("orientations and times length mismatch")
        if any(t_next < t_curr for t_curr, t_next in zip(self.times, self.times[1:])):
            raise ValueError("times must be non-decreasing")

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if self.times else 0.0

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        if not self.times:
            raise ValueError("Empty trajectory")
        if t <= self.times[0]:
            return np.asarray(self.positions[0], dtype=np.float64), np.asarray(self.orientations[0], dtype=np.float64)
        if t >= self.times[-1]:
            return np.asarray(self.positions[-1], dtype=np.float64), np.asarray(self.orientations[-1], dtype=np.float64)

        idx = bisect_left(self.times, t)
        idx = min(max(idx, 1), len(self.times) - 1)
        t0, t1 = self.times[idx - 1], self.times[idx]
        span = t1 - t0
        alpha = 0.0 if span <= 1e-9 else (t - t0) / span
        pos0 = np.asarray(self.positions[idx - 1], dtype=np.float64)
        pos1 = np.asarray(self.positions[idx], dtype=np.float64)
        ori0 = np.asarray(self.orientations[idx - 1], dtype=np.float64)
        ori1 = np.asarray(self.orientations[idx], dtype=np.float64)
        position = (1.0 - alpha) * pos0 + alpha * pos1
        orientation = (1.0 - alpha) * ori0 + alpha * ori1
        norm = np.linalg.norm(orientation)
        if norm > 0:
            orientation = orientation / norm
        return position, orientation


class TrajectoryInterpolator:
    """Utility for building interpolated trajectories."""

    @staticmethod
    def linear(points: Sequence[TrajectoryPoint], joint_names: Sequence[str] | None = None) -> JointTrajectory:
        if not points:
            raise ValueError("points must not be empty")
        joint_count = len(points[0].positions)
        names = tuple(joint_names) if joint_names is not None else tuple(f"j{idx}" for idx in range(joint_count))
        times = tuple(float(pt.time) for pt in points)
        positions = tuple(tuple(float(v) for v in pt.positions) for pt in points)
        velocities = (
            tuple(tuple(float(v) for v in pt.velocities) for pt in points)
            if all(pt.velocities is not None for pt in points)
            else None
        )
        accelerations = (
            tuple(tuple(float(v) for v in pt.accelerations) for pt in points)
            if all(pt.accelerations is not None for pt in points)
            else None
        )
        return JointTrajectory(names, times, positions, velocities, accelerations)

    @staticmethod
    def cubic(points: Sequence[TrajectoryPoint], joint_names: Sequence[str] | None = None) -> JointTrajectory:
        return TrajectoryInterpolator.linear(points, joint_names)

    @staticmethod
    def from_waypoints(
        waypoints: np.ndarray,
        times: Sequence[float],
        velocities: Optional[np.ndarray] = None,
        accelerations: Optional[np.ndarray] = None,
        joint_names: Sequence[str] | None = None,
    ) -> JointTrajectory:
        if waypoints.shape[0] != len(times):
            raise ValueError("times length must match number of waypoints")
        points = []
        for idx, t in enumerate(times):
            vel = velocities[idx] if velocities is not None else None
            acc = accelerations[idx] if accelerations is not None else None
            points.append(
                TrajectoryPoint(
                    positions=tuple(float(v) for v in waypoints[idx]),
                    velocities=tuple(float(v) for v in vel) if vel is not None else None,
                    accelerations=tuple(float(v) for v in acc) if acc is not None else None,
                    time=float(t),
                )
            )
        return TrajectoryInterpolator.linear(points, joint_names)


def make_trajectory(
    joint_names: Sequence[str],
    times_s: Sequence[float],
    positions: Sequence[Sequence[float]],
    velocities: Optional[Sequence[Sequence[float]]] = None,
    accelerations: Optional[Sequence[Sequence[float]]] = None,
) -> JointTrajectory:
    if len(times_s) != len(positions):
        raise ValueError("times and positions length mismatch")
    if velocities is not None and len(velocities) != len(times_s):
        raise ValueError("times and velocities length mismatch")
    if accelerations is not None and len(accelerations) != len(times_s):
        raise ValueError("times and accelerations length mismatch")

    indices = sorted(range(len(times_s)), key=lambda idx: times_s[idx])
    sorted_times: list[float] = []
    sorted_positions: list[Tuple[float, ...]] = []
    sorted_velocities: list[Tuple[float, ...]] | None = [] if velocities is not None else None
    sorted_accelerations: list[Tuple[float, ...]] | None = [] if accelerations is not None else None

    for idx in indices:
        sorted_times.append(float(times_s[idx]))
        sorted_positions.append(tuple(float(v) for v in positions[idx]))
        if velocities is not None and sorted_velocities is not None:
            sorted_velocities.append(tuple(float(v) for v in velocities[idx]))
        if accelerations is not None and sorted_accelerations is not None:
            sorted_accelerations.append(tuple(float(v) for v in accelerations[idx]))

    if sorted_times and sorted_times[0] != 0.0:
        sorted_times.insert(0, 0.0)
        sorted_positions.insert(0, sorted_positions[0])
        if sorted_velocities is not None:
            sorted_velocities.insert(0, sorted_velocities[0])
        if sorted_accelerations is not None:
            sorted_accelerations.insert(0, sorted_accelerations[0])

    velocities_tuple = tuple(sorted_velocities) if sorted_velocities is not None else None
    accelerations_tuple = tuple(sorted_accelerations) if sorted_accelerations is not None else None

    return JointTrajectory(
        joint_names=tuple(joint_names),
        times_s=tuple(sorted_times),
        positions=tuple(sorted_positions),
        velocities=velocities_tuple,
        accelerations=accelerations_tuple,
    )


def densify_waypoints(
    waypoints: Sequence[Sequence[float]],
    max_joint_step: float = 0.02,
) -> Tuple[Tuple[float, ...], ...]:
    dense: list[list[float]] = []
    for idx, waypoint in enumerate(waypoints):
        if idx == 0:
            dense.append(list(waypoint))
            continue
        prev = np.asarray(waypoints[idx - 1], dtype=np.float64)
        nxt = np.asarray(waypoint, dtype=np.float64)
        diff = nxt - prev
        span = float(np.max(np.abs(diff)))
        steps = max(1, int(np.ceil(span / max_joint_step)))
        for s in range(1, steps + 1):
            alpha = s / steps
            interp = prev + alpha * diff
            dense.append(interp.tolist())
    return tuple(tuple(float(v) for v in wp) for wp in dense)


__all__ = [
    "TrajectoryPoint",
    "TrajectorySegment",
    "JointTrajectory",
    "CartesianTrajectory",
    "TrajectoryInterpolator",
    "make_trajectory",
    "densify_waypoints",
]
