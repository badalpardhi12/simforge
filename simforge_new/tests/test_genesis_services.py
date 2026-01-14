"""Unit tests for Genesis-backed IK, planning, and collision helpers."""
from __future__ import annotations

import logging
from types import SimpleNamespace


import numpy as np
import pytest
import torch

import genesis as gs

from simforge_new.core.models import Pose, RobotProfile, MountPose, MotionTarget
from simforge_new.core.config_schema import RobotControlPolicy, SafetyPolicy, PlannerStrategy
from simforge_new.services import (
    GenesisIKSolver,
    GenesisMotionPlanner,
    GenesisCollisionWorld,
    PlanRequest,
    PlanOutcome,
    IKRequest,
    CollisionQuery,
)


@pytest.fixture(scope="module", autouse=True)
def _init_genesis():
    try:
        gs.init(backend=gs.cpu)
    except Exception as exc:  # pragma: no cover - already initialised
        if "already initialized" not in str(exc).lower():
            raise
    yield


class _StubEntity:
    def __init__(self, *, dof: int = 6, link_name: str = "tool0") -> None:
        self.n_qs = dof
        self.links = [SimpleNamespace(name=link_name)]
        self._qpos = torch.zeros(dof, dtype=gs.tc_float, device=gs.device)

    # IK helpers
    def get_link(self, name=None, uid=None):
        if name is None:
            return self.links[-1]
        for link in self.links:
            if link.name == name:
                return link
        raise KeyError(name)

    def get_qpos(self):
        return self._qpos.clone()

    def set_qpos(self, tensor, zero_velocity=True):
        self._qpos = tensor.clone()

    def inverse_kinematics(self, *, init_qpos, **kwargs):
        # Return the provided seed as the solution with zero error.
        return init_qpos, torch.zeros((init_qpos.shape[0], 6), dtype=gs.tc_float, device=gs.device)

    # Planner helpers
    def plan_path(self, qpos_goal, qpos_start=None, **_):
        start = qpos_start if qpos_start.ndim == 1 else qpos_start[0]
        goal = qpos_goal if qpos_goal.ndim == 1 else qpos_goal[0]
        mid = (start + goal) * 0.5
        path = torch.stack([start, mid, goal], dim=0).unsqueeze(1)
        return path, torch.tensor(True)

    # Collision helpers
    def detect_collision(self):
        if torch.any(self._qpos > 0.5):
            return np.ones((1, 2), dtype=np.int32)
        return np.zeros((0, 2), dtype=np.int32)


class _StubContext:
    def __init__(self, *, profile, entity, joint_count, logger, safety_policy) -> None:
        self.profile = profile
        self.entity = entity
        self.joint_count = joint_count
        self.logger = logger
        self.safety_policy = safety_policy


def _make_profile(name: str = "robot") -> RobotProfile:
    mount = MountPose(position=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))
    return RobotProfile(
        name=name,
        urdf="dummy",
        mount=mount,
        fixed_base=True,
        control=RobotControlPolicy(),
        end_effector_link="tool0",
        metadata={},
    )


def test_genesis_ik_solver_success() -> None:
    profile = _make_profile()
    entity = _StubEntity()
    solver = GenesisIKSolver(profile=profile, entity=entity)
    target = MotionTarget(pose=Pose(position=(0.0, 0.0, 0.1), orientation=(1.0, 0.0, 0.0, 0.0)))
    request = IKRequest(robot=profile, target=target, max_attempts=1)

    result = solver.solve(request)

    assert result.success
    assert result.solution is not None
    assert isinstance(result.metrics.position_error_m, float)


def test_genesis_motion_planner_success() -> None:
    profile = _make_profile()
    entity = _StubEntity()
    planner = GenesisMotionPlanner(profile=profile, entity=entity, collision_world=None)

    request = PlanRequest(
        robot=profile,
        start=(0.0,) * entity.n_qs,
        goal=(0.1,) * entity.n_qs,
        strategy=PlannerStrategy.JOINT_ONLY,
        timeout_s=2.0,
    )

    result = planner.plan(request)

    assert result.outcome == PlanOutcome.SUCCESS
    assert result.trajectory is not None
    assert len(result.trajectory.positions) == 3


def test_genesis_motion_planner_validator_rejects() -> None:
    profile = _make_profile()
    entity = _StubEntity()
    planner = GenesisMotionPlanner(profile=profile, entity=entity, collision_world=None)

    def _validator(joints: tuple[float, ...]) -> bool:
        total = sum(joints)
        return abs(total - 0.3) > 1e-6

    request = PlanRequest(
        robot=profile,
        start=(0.0,) * entity.n_qs,
        goal=(0.1,) * entity.n_qs,
        strategy=PlannerStrategy.JOINT_ONLY,
        timeout_s=2.0,
        is_state_valid=_validator,
    )

    result = planner.plan(request)

    assert result.outcome == PlanOutcome.COLLISION
    assert result.trajectory is None


def test_genesis_collision_world_detects_collision() -> None:
    profile = _make_profile()
    entity = _StubEntity()
    ctx = _StubContext(
        profile=profile,
        entity=entity,
        joint_count=entity.n_qs,
        logger=logging.getLogger("robot.ctx"),
        safety_policy=SafetyPolicy(),
    )

    world = GenesisCollisionWorld({profile.name: ctx}, safety=SafetyPolicy())

    query = CollisionQuery(robot=profile, joints=(0.6,) * entity.n_qs, other_robot_states={})
    result = world.is_state_valid(query)

    assert result.in_collision
    assert result.distance_m == 0.0
