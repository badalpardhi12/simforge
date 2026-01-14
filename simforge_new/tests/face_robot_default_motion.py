"""Utility to batch-test Genesis IK/planning for the face_robot preset."""
from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

# Ensure repository root is on sys.path when the file is executed directly
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simforge_new.control.session import SimulationSession
from simforge_new.core import Backend, CartesianMoveCommand, PlannerStrategy
from simforge_new.core.models import RobotProfile
from simforge_new.services.ik.base import IKRequest
from simforge_new.services.planning.base import PlanRequest, PlanOutcome


@dataclass
class PoseResult:
    position_m: Tuple[float, float, float]
    orientation_deg: Tuple[float, float, float]
    ik_success: bool
    plan_success: bool
    ik_error_m: float
    ik_error_deg: float
    failure_reason: str | None = None


HORIZ_SHIFT_M = tuple(round(v, 3) for v in np.linspace(-0.1, 0.1, 5))
VERT_SHIFT_M = tuple(round(v, 3) for v in np.linspace(-0.1, 0.1, 5))
DEFAULT_DISTANCE_M = 0.3  # 300 mm in GUI
DEFAULT_RPY_DEG = (90.0, 0.0, 0.0)
REFERENCE_FRAME = "obj:face_object_0"


async def _evaluate_face_robot_grid() -> List[PoseResult]:
    session = await SimulationSession.create(
        "simforge_new/environment/presets/face_robot.yaml",
        backend=Backend.GPU,
    )
    results: List[PoseResult] = []
    try:
        coordinator = next(iter(session._coordinators.values()))
        robot: RobotProfile = coordinator.ctx.profile
        solver = coordinator.ik_solver
        planner = coordinator.planner
        validator = coordinator._state_validator()

        for dx in HORIZ_SHIFT_M:
            for dz in VERT_SHIFT_M:
                cmd = CartesianMoveCommand(
                    command_id=f"proto_{dx}_{dz}",
                    position_m=(dx, 0.3, dz),
                    orientation_deg=DEFAULT_RPY_DEG,
                    duration=4.0,
                    reference_frame=REFERENCE_FRAME,
                    metadata={},
                    priority=0,
                )
                target = coordinator._motion_target_from_cartesian(cmd)
                seed = tuple(float(v) for v in coordinator._current_joints(coordinator.ctx.joint_count))

                ik_req = IKRequest(
                    robot=robot,
                    target=target,
                    seed=seed,
                    position_tolerance_m=coordinator._ik_pos_tol,
                    orientation_tolerance_deg=coordinator._ik_rot_tol,
                    is_state_valid=validator,
                    max_attempts=8,
                )

                ik_result = solver.solve(ik_req)
                ik_success = bool(ik_result.success and ik_result.solution)
                pos_err = math.nan
                ang_err = math.nan
                failure_reason = None

                if ik_success:
                    joints = np.asarray(ik_result.solution, dtype=np.float64)
                    pos_err, ang_err = coordinator._compute_pose_error(joints, target.pose)
                else:
                    failure_reason = str((ik_result.raw or {}).get("reason", "ik_failure"))

                plan_success = False
                if ik_success:
                    plan_req = PlanRequest(
                        robot=robot,
                        start=seed,
                        goal=ik_result.solution,
                        strategy=PlannerStrategy.CARTESIAN_PREFERRED,
                        timeout_s=3.0,
                        is_state_valid=validator,
                    )
                    plan = planner.plan(plan_req)
                    plan_success = plan.trajectory is not None and plan.outcome == PlanOutcome.SUCCESS
                    if not plan_success:
                        failure_reason = f"planner_{(plan.outcome.value if plan.outcome else 'failure')}"

                results.append(
                    PoseResult(
                        position_m=cmd.position_m,
                        orientation_deg=DEFAULT_RPY_DEG,
                        ik_success=ik_success,
                        plan_success=plan_success,
                        ik_error_m=float(pos_err) if math.isfinite(pos_err) else float("nan"),
                        ik_error_deg=float(ang_err) if math.isfinite(ang_err) else float("nan"),
                        failure_reason=failure_reason,
                    )
                )
        return results
    finally:
        await session.close()


def run_default_face_robot_grid() -> List[PoseResult]:
    """Synchronous helper for CLI or tests."""
    return asyncio.run(_evaluate_face_robot_grid())


def main() -> None:
    results = run_default_face_robot_grid()
    successes = [r for r in results if r.ik_success and r.plan_success]
    failures = [r for r in results if not (r.ik_success and r.plan_success)]
    print(f"Evaluated {len(results)} face_robot poses: {len(successes)} success, {len(failures)} failure")
    for entry in successes:
        print(
            f"  ✔ {entry.position_m}: pos_err={entry.ik_error_m:.4f} m, "
            f"rot_err={entry.ik_error_deg:.3f}°"
        )
    for entry in failures:
        print(
            f"  ✖ {entry.position_m}: reason={entry.failure_reason} "
            f"| ik_err=({entry.ik_error_m:.4f} m, {entry.ik_error_deg:.3f}°)"
        )


if __name__ == "__main__":
    main()
