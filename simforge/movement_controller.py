"""Backward-compatible shim exposing the reorganized controller package."""
from __future__ import annotations

from .controller import (
    MovementController,
    ControlMode,
    Command,
    SetJointCommand,
    SetJointTargetsCommand,
    CartesianMoveCommand,
    SwitchModeCommand,
)

__all__ = [
    "MovementController",
    "ControlMode",
    "Command",
    "SetJointCommand",
    "SetJointTargetsCommand",
    "CartesianMoveCommand",
    "SwitchModeCommand",
]
