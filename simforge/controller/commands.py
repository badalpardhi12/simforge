"""Command types for the movement controller."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class ControlMode(Enum):
    """Supported control modes for a robot."""
    JOINT = "joint"
    CARTESIAN = "cartesian"


@dataclass
class Command:
    """Base command type."""
    pass


@dataclass
class SetJointCommand(Command):
    robot: str
    joint_idx: int
    value_deg: float


@dataclass
class SetJointTargetsCommand(Command):
    robot: str
    values_deg: List[float]


@dataclass
class CartesianMoveCommand(Command):
    robot: str
    position: Tuple[float, float, float]
    orientation_deg: Tuple[float, float, float]  # roll, pitch, yaw
    frame: str = "base"


@dataclass
class SwitchModeCommand(Command):
    robot: str
    mode: ControlMode


__all__ = [
    "ControlMode",
    "Command",
    "SetJointCommand",
    "SetJointTargetsCommand",
    "CartesianMoveCommand",
    "SwitchModeCommand",
]
