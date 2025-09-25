"""Unit tests for trajectory synchronization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from simforge_new.control.synchronized_trajectory import SynchronizedTrajectoryManager
from simforge_new.core.trajectories import JointTrajectory


ROBOT = "arm"


def _linear_trajectory() -> JointTrajectory:
    return JointTrajectory(
        joint_names=("j0", "j1"),
        times_s=(0.0, 0.5, 1.0),
        positions=(
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
        ),
    )


@pytest.fixture
def manager() -> SynchronizedTrajectoryManager:
    return SynchronizedTrajectoryManager()


def test_start_and_sample(monkeypatch: pytest.MonkeyPatch, manager: SynchronizedTrajectoryManager) -> None:
    start_time = 100.0
    monkeypatch.setattr("simforge_new.control.synchronized_trajectory.time.time", lambda: start_time)

    trajectory = _linear_trajectory()
    manager.start_trajectory(ROBOT, trajectory)

    assert manager.has_active_trajectory(ROBOT)
    assert set(manager.get_active_robots()) == {ROBOT}

    current = start_time + 0.25
    sample = manager.get_current_positions(ROBOT, current)
    assert sample is not None
    assert np.allclose(sample, np.array([0.25, 0.25]))


def test_completion_moves_to_hold(monkeypatch: pytest.MonkeyPatch, manager: SynchronizedTrajectoryManager) -> None:
    start_time = 50.0
    monkeypatch.setattr("simforge_new.control.synchronized_trajectory.time.time", lambda: start_time)

    manager.start_trajectory(ROBOT, _linear_trajectory())

    completion_time = start_time + 2.0
    assert manager.is_trajectory_done(ROBOT, completion_time)
    assert not manager.has_active_trajectory(ROBOT)

    held = manager.get_current_positions(ROBOT, completion_time + 0.1)
    assert held is not None
    assert np.allclose(held, np.array([1.0, 1.0]))


def test_hold_position_overridden_by_new_trajectory(
    monkeypatch: pytest.MonkeyPatch, manager: SynchronizedTrajectoryManager
) -> None:
    manual_hold = np.array([0.3, -0.2])
    manager.hold_position(ROBOT, manual_hold)

    still = manager.get_current_positions(ROBOT, current_time=0.0)
    assert still is not None
    assert np.allclose(still, manual_hold)

    start_time = 10.0
    monkeypatch.setattr("simforge_new.control.synchronized_trajectory.time.time", lambda: start_time)
    manager.start_trajectory(ROBOT, _linear_trajectory())

    assert manager.has_active_trajectory(ROBOT)
    first_sample = manager.get_current_positions(ROBOT, start_time)
    assert first_sample is not None
    assert np.allclose(first_sample, np.array([0.0, 0.0]))
