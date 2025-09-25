"""Face robot interaction demo leveraging the Simforge stack."""
from __future__ import annotations

import asyncio
from importlib import resources
from typing import Iterable

import numpy as np

from ..control.session import SimulationSession
from ..core.commands import CartesianMoveCommand, JointTargetsCommand

DESCRIPTION = "TX2-90XL performing the face interaction pose tour"


async def _send_joint_home(
    session: SimulationSession,
    robot_name: str,
    target_deg: Iterable[float],
    *,
    duration_s: float = 3.0,
    tolerance_deg: float = 0.5,
    max_attempts: int = 3,
) -> None:
    target_deg_list = list(target_deg)
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        command = JointTargetsCommand(
            robot_name=robot_name,
            values_deg=target_deg_list,
            duration=duration_s,
            metadata={
                "source": "demo_home",
            },
        )
        await session.send_command(command)
        timeout = max(duration_s * 2.0, 15.0)
        await session.wait_until_idle(robot_name, timeout=timeout)

        state = await session.wait_for_robot_state(robot_name)
        actual_deg = np.rad2deg(state.joint_positions)
        max_err = float(np.max(np.abs(actual_deg - target_deg_list)))
        if max_err <= tolerance_deg:
            return

        print(
            f"  ⚠ Home pose off by {max_err:.2f}° (attempt {attempts}/{max_attempts});"
            " retrying…"
        )
        await asyncio.sleep(0.5)

    raise RuntimeError("home_pose_not_reached")


async def run_demo() -> None:
    """Execute the face interaction demo sequence."""

    print("=" * 60)
    print("Face Robot Demo – Simforge")
    print("=" * 60)

    config_resource = resources.files("simforge_new.environment.presets") / "face_robot.yaml"
    config_path = str(config_resource)

    session = await SimulationSession.create(config_path)

    robot_names = list(session.spec.robot_names)
    if not robot_names:
        raise RuntimeError("Environment did not load any robots")

    robot_name = robot_names[0]
    profile = next(robot for robot in session.spec.robots if robot.name == robot_name)

    print(f"\n✓ Environment loaded from {config_resource.name}")
    print(f"  Robots: {', '.join(robot_names)}")

    initial_state = await session.wait_for_robot_state(robot_name)
    print(f"  Initial joint positions (deg): {np.round(np.rad2deg(initial_state.joint_positions), 2)}")

    home_deg = (
        list(profile.initial_joint_positions_deg)
        if profile.initial_joint_positions_deg
        else [0.0] * profile.joint_count
    )

    print("\nResetting to home pose…")
    await _send_joint_home(session, robot_name, home_deg)

    poses = [
        {"pos": (0.2, 0.2, 0.2), "rpy": (90.0, 0.0, 0.0)},
        {"pos": (-0.2, 0.2, 0.2), "rpy": (90.0, 0.0, 0.0)},
        {"pos": (-0.2, 0.2, -0.2), "rpy": (90.0, 0.0, 0.0)},
        {"pos": (0.2, 0.2, -0.2), "rpy": (90.0, 0.0, 0.0)},
    ]

    # frame = "obj:face_object_0"
    # run on 8 face_object poses
    frames = ["obj:face_object_0", 
              "obj:face_object_1", 
              "obj:face_object_2", 
              "obj:face_object_3",
              "obj:face_object_4",
              "obj:face_object_5",
              "obj:face_object_6",
              "obj:face_object_7"]
    max_retries = 2

    # print("\nStarting face_object_0 sequence…\n" + "-" * 40)
    for frame in frames:
        print(f"\nStarting sequence relative to {frame}…\n" + "-" * 40)
        for pose_idx, pose in enumerate(poses, start=1):
            target_desc = f"Pose {pose_idx}: pos={pose['pos']}, rpy={pose['rpy']}"
            print(target_desc)

            attempt = 0
            while attempt <= max_retries:
                if attempt:
                    print(f"  Retry {attempt + 1}/{max_retries + 1}…")

                command = CartesianMoveCommand(
                    robot_name=robot_name,
                    position_m=list(pose["pos"]),
                    orientation_deg=list(pose["rpy"]),
                    duration=4.0,
                    reference_frame=frame,
                    use_collision_checking=True,
                )

                await session.send_command(command)

                try:
                    await session.wait_until_idle(robot_name, timeout=30.0)
                except TimeoutError as exc:
                    print(f"  ✗ Timed out waiting for completion: {exc}")
                    attempt += 1
                    if attempt > max_retries:
                        raise RuntimeError("cartesian_move_timeout") from exc
                    await asyncio.sleep(0.5)
                    continue

                latest = await session.wait_for_robot_state(robot_name)
                print(f"  → Current joints (deg): {np.round(np.rad2deg(latest.joint_positions), 2)}")
                await asyncio.sleep(0.5)
                break

    print("\nSequence complete. Returning to home pose…")
    await _send_joint_home(session, robot_name, home_deg)

    final_state = await session.wait_for_robot_state(robot_name)
    print(f"  Final joints (deg): {np.round(np.rad2deg(final_state.joint_positions), 2)}")

    print("\nShutting down session…")
    await session.close()
    print("✓ Demo complete")


__all__ = ["run_demo", "DESCRIPTION"]
