#!/usr/bin/env python3
"""
Automated proto-sim test for face_link target with mode switching.

Flow:
1. Connect to simforge command gateway.
2. Fetch environment info and face_link transform.
3. Generate poses on the client side (same path as GUI).
4. Run protocol in simulation mode.
5. Switch to real mode ("both") and run the same protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# Allow running as a script from repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simforge_client import SimforgeClient
from simforge_client.utils.pose_generation import ProtoSimParameters, generate_world_poses


TEST_PARAMS = ProtoSimParameters(
    horiz=[0.0],
    vert=[0.0],
    distance=[350.0],
    roll=[-90.0],
    pitch=[15.0],
    yaw=[-30.0, 0.0, 30.0],
)


def _print_json(title: str, payload: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print(json.dumps(payload, indent=2, sort_keys=False))


async def _prepare_mode(client: SimforgeClient, mode: str) -> Dict[str, Any]:
    response = await client.call_rpc("prepare_mode", {"mode": mode}, timeout=180.0)
    _print_json(f"prepare_mode({mode})", response)
    if not response.get("ready", False):
        raise RuntimeError(
            f"prepare_mode('{mode}') failed: {response.get('message', 'unknown error')}"
        )
    return response


async def _run_proto(
    client: SimforgeClient,
    mode: str,
    robot_name: str,
    poses_data: List[Dict[str, Any]],
    idle_time: float,
    move_speed: float,
) -> Dict[str, Any]:
    # Conservative timeout for mode switching + planning + execution + return-home.
    timeout = max(240.0, len(poses_data) * (idle_time + 20.0) + 120.0)
    payload = {
        "robot_name": robot_name,
        "poses": poses_data,
        "idle_time": idle_time,
        "mode": mode,
        "move_speed": move_speed,
    }
    response = await client.call_rpc("run_proto_sim", payload, timeout=timeout)
    _print_json(f"run_proto_sim(mode={mode})", response)
    if not response.get("success", False):
        raise RuntimeError(
            f"run_proto_sim(mode={mode}) failed: {response.get('message', 'unknown error')}"
        )
    return response


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run face_link protocol in simulation first, then real mode."
    )
    parser.add_argument("--server", default="127.0.0.1", help="Command gateway host/IP")
    parser.add_argument("--port", type=int, default=8766, help="Command gateway port")
    parser.add_argument(
        "--robot",
        default="nakul_ur5e",
        help="Robot name (e.g., nakul_ur5e or sahadev_ur5e)",
    )
    parser.add_argument("--target-object", default="face_link", help="Target object frame name")
    parser.add_argument("--idle-time", type=float, default=2.0, help="Idle time per pose (s)")
    parser.add_argument(
        "--move-speed",
        type=float,
        default=0.2,
        help="Velocity scaling hint sent to server (0.0-1.0)",
    )
    parser.add_argument(
        "--real-mode-name",
        default="both",
        help="Mode string for real robot path (default: both)",
    )
    parser.add_argument(
        "--allow-legacy-backend",
        action="store_true",
        help="Allow execution even if server does not look like NVIDIA cuMotion backend.",
    )
    args = parser.parse_args()

    print(f"Connecting to ws://{args.server}:{args.port}")
    async with SimforgeClient(
        server_ip=args.server,
        command_port=args.port,
        client_id="face_link_mode_switch_test",
    ) as client:
        print("Connected to command gateway")

        env_info = await client.call_rpc("get_environment_info", {}, timeout=15.0)
        _print_json("get_environment_info", env_info)
        if not env_info.get("success", False):
            raise RuntimeError(
                f"get_environment_info failed: {env_info.get('error', 'unknown error')}"
            )

        robots = env_info.get("robots", [])
        if args.robot not in robots:
            if not robots:
                raise RuntimeError("No robots reported by server")
            print(
                f"Requested robot '{args.robot}' not found, using '{robots[0]}' instead"
            )
            args.robot = robots[0]

        object_transforms = env_info.get("object_transforms", {})
        tf = object_transforms.get(args.target_object)
        if not tf:
            raise RuntimeError(
                f"Transform for target object '{args.target_object}' is missing in get_environment_info"
            )

        target_position = tuple(tf["position"])
        target_orientation = tuple(tf["orientation"])

        world_poses = generate_world_poses(
            TEST_PARAMS,
            target_position=target_position,
            target_orientation_quat_xyzw=target_orientation,
            randomize=False,
        )
        poses_data = [p.to_dict() for p in world_poses]

        print(f"\nGenerated {len(poses_data)} poses for {args.target_object}:")
        for pose in poses_data:
            pos = pose["position"]
            quat = pose["orientation"]
            print(
                f"  {pose['name']}: "
                f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) "
                f"quat=({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})"
            )

        status = await client.call_rpc("get_robot_status", {"robot_name": args.robot}, timeout=10.0)
        _print_json(f"get_robot_status({args.robot})", status)

        connection_details = status.get("connection_details", {})
        has_cumotion = "cumotion_motion_plan_action" in connection_details
        reference_frame = env_info.get("reference_frame")
        looks_like_nvidia = has_cumotion and reference_frame == "world"
        if not looks_like_nvidia and not args.allow_legacy_backend:
            raise RuntimeError(
                "Connected gateway does not appear to be the NVIDIA backend "
                f"(reference_frame={reference_frame!r}, "
                f"has_cumotion_motion_plan_action={has_cumotion}). "
                "Start simforge_server_nvidia or pass --allow-legacy-backend."
            )

        # 1) Simulation run
        await _prepare_mode(client, "simulation")
        await _run_proto(
            client,
            mode="simulation",
            robot_name=args.robot,
            poses_data=poses_data,
            idle_time=args.idle_time,
            move_speed=args.move_speed,
        )

        # 2) Real run
        await _prepare_mode(client, args.real_mode_name)
        await _run_proto(
            client,
            mode=args.real_mode_name,
            robot_name=args.robot,
            poses_data=poses_data,
            idle_time=args.idle_time,
            move_speed=args.move_speed,
        )

    print("\nTest completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
