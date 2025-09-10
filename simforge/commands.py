from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SetJointDeg:
    robot: str
    idx: int
    val_deg: float


@dataclass
class SetJointTargetsDeg:
    robot: str
    vals_deg: List[float]


@dataclass
class SwitchMode:
    robot: str
    mode: str


@dataclass
class CartesianMove:
    robot: str
    pose_xyzrpy: Tuple[float, float, float, float, float, float]
    frame: str


@dataclass
class CartesianCancel:
    robot: str


Command = (
    SetJointDeg
    | SetJointTargetsDeg
    | SwitchMode
    | CartesianMove
    | CartesianCancel
)