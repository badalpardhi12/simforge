"""Collision checking service contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Protocol, Tuple

from ...core.models import RobotProfile

JointVector = Tuple[float, ...]


@dataclass(frozen=True)
class CollisionQuery:
    robot: RobotProfile
    joints: JointVector
    other_robot_states: Dict[str, JointVector]

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", tuple(float(v) for v in self.joints))
        object.__setattr__(self, "other_robot_states", {
            key: tuple(float(x) for x in value) for key, value in (self.other_robot_states or {}).items()
        })


@dataclass(frozen=True)
class CollisionCheck:
    distance_m: float
    in_collision: bool
    details: Dict[str, object] = None

    def __post_init__(self) -> None:
        if self.details is None:
            object.__setattr__(self, "details", {})


class CollisionWorld(Protocol):
    """Protocol implemented by collision backends."""

    def is_state_valid(self, query: CollisionQuery) -> CollisionCheck:  # pragma: no cover
        ...

    def update_environment(self, robot_states: Dict[str, JointVector]) -> None:  # pragma: no cover
        ...

    def allowed_pairs(self) -> Iterable[Tuple[str, str]]:  # pragma: no cover
        ...


__all__ = ["CollisionQuery", "CollisionCheck", "CollisionWorld"]
