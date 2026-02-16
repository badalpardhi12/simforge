"""Lightweight state estimator that polls Genesis entities."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional
import threading

import numpy as np

from .base import StateEstimator, StateSubscription
from ...core.models import RobotProfile, RobotState
from ...infrastructure.genesis.io import get_joint_positions


class PollingStateEstimator(StateEstimator):
    """Polls robot DOFs at a fixed frequency and yields :class:`RobotState`."""

    def __init__(
        self,
        entities: Dict[str, object],
        profiles: Dict[str, RobotProfile],
        *,
        default_frequency_hz: float = 60.0,
        logger: Optional[logging.Logger] = None,
        lock: Optional[threading.RLock] = None,
    ) -> None:
        self._entities = dict(entities)
        self._profiles = dict(profiles)
        self._default_frequency = float(default_frequency_hz)
        self._logger = logger or logging.getLogger("simforge.state.polling")
        self._references: Dict[str, np.ndarray] = {}
        self._lock = lock or threading.RLock()

    async def states(self, subscription: StateSubscription):
        robot = subscription.robot.name
        entity = self._entities.get(robot)
        if entity is None:
            raise KeyError(f"Robot '{robot}' not registered with PollingStateEstimator")
        profile = self._profiles.get(robot)
        if profile is None:
            raise KeyError(f"Robot profile for '{robot}' missing")

        frequency = subscription.frequency_hz or self._default_frequency
        dt = 1.0 / frequency if frequency > 0 else 1.0 / max(self._default_frequency, 1.0)

        try:
            while True:
                with self._lock:
                    joints = get_joint_positions(entity, self._logger, prefer_struct=True)
                joints = self._resize(robot, joints, profile)
                timestamp = time.time()
                state = RobotState(
                    name=robot,
                    joint_positions=tuple(float(v) for v in joints),
                    joint_velocities=tuple(0.0 for _ in range(joints.size)),
                    timestamp_s=timestamp,
                    frame=profile.control.mode,
                )
                yield state
                await asyncio.sleep(dt)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            raise

    async def set_reference(self, robot: RobotProfile, joints):
        data = np.asarray(tuple(float(v) for v in joints), dtype=np.float64)
        self._references[robot.name] = data

    def _resize(self, robot: str, joints: np.ndarray, profile: RobotProfile) -> np.ndarray:
        if joints.size:
            return joints
        ref = self._references.get(robot)
        if ref is not None:
            return ref
        if profile.initial_joint_positions_deg:
            return np.deg2rad(np.asarray(profile.initial_joint_positions_deg, dtype=np.float64))
        dof = int(profile.metadata.get("dof", 6)) if profile.metadata else 6
        return np.zeros(dof, dtype=np.float64)


__all__ = ["PollingStateEstimator"]
