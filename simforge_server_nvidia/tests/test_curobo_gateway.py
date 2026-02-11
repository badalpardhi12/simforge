#!/usr/bin/env python3
"""
Test script for NVIDIA cuRobo gateway.

Supports exclusive sim-only or real-only modes (default: sim-only).

Replicates the face_robot_demo workflow:
  1. Connect to cuRobo gateway on ws://<server>:8766
  2. Get environment info (face_link TF)
  3. Generate spherical poses: horiz=0, vert=0, dist=350, roll=-90,
     pitch=15, yaw=[-30, 0, 30]
  4. Prepare the requested mode (simulation or real)
  5. Execute the trajectory

Usage:
  # Simulation test (default)
  python test_curobo_gateway.py --server <IP>

  # Simulation test (explicit)
  python test_curobo_gateway.py --server <IP> --sim-only

  # Real robot test (uses RTDE — bypasses ROS2 control stack)
  python test_curobo_gateway.py --server <IP> --real-only --speed 0.2
"""

import argparse
import asyncio
import json
import math
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    import websockets
except ImportError:
    print("pip install websockets>=12.0")
    sys.exit(1)


# ── Inline pose generation (matches simforge_client/utils/pose_generation.py) ──

def _mm_to_m(v: float) -> float:
    return v / 1000.0


def _spherical_to_cartesian(
    pitch_deg: float, yaw_deg: float, dist_mm: float
) -> Tuple[float, float, float]:
    d = _mm_to_m(dist_mm)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (sy * d * cp, cy * d * cp, d * sp)


def _quat_from_matrix(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    else:
        diag = [m00, m11, m22]
        idx = max(range(3), key=lambda i: diag[i])
        if idx == 0:
            s = math.sqrt(1 + m00 - m11 - m22) * 2
            x, w = 0.25 * s, (m21 - m12) / s
            y, z = (m01 + m10) / s, (m02 + m20) / s
        elif idx == 1:
            s = math.sqrt(1 + m11 - m00 - m22) * 2
            y, w = 0.25 * s, (m02 - m20) / s
            x, z = (m01 + m10) / s, (m12 + m21) / s
        else:
            s = math.sqrt(1 + m22 - m00 - m11) * 2
            z, w = 0.25 * s, (m10 - m01) / s
            x, y = (m02 + m20) / s, (m12 + m21) / s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def _rpy_to_quat_xyzw(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (
        sr * cp * cy + cr * sp * sy,
        cr * sp * cy - sr * cp * sy,
        cr * cp * sy + sr * sp * cy,
        cr * cp * cy - sr * sp * sy,
    )


def _quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _normalize_quat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n) if n > 1e-12 else (0, 0, 0, 1)


def _look_at_quaternion(pos, target, roll_deg=0.0):
    import numpy as np

    src = np.asarray(pos, dtype=float)
    tgt = np.asarray(target, dtype=float)
    y_axis = tgt - src
    n = np.linalg.norm(y_axis)
    y_axis = y_axis / n if n > 1e-9 else np.array([0, 1, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, y_axis)) >= 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(y_axis, up)
    xn = np.linalg.norm(x_axis)
    x_axis = x_axis / xn if xn > 1e-9 else np.array([1.0, 0.0, 0.0])
    z_axis = np.cross(x_axis, y_axis)
    rot = np.column_stack((x_axis, y_axis, z_axis))
    q = _quat_from_matrix(rot.tolist())
    if abs(roll_deg) > 1e-6:
        rq = _rpy_to_quat_xyzw(0, math.radians(roll_deg), 0)
        q = _quat_multiply(q, rq)
    return _normalize_quat(q)


def _quat_rot_point(quat_xyzw, point):
    """Rotate a 3D point by quaternion (x,y,z,w)."""
    import numpy as np

    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ]
    p = np.array(point)
    r = np.array(rot)
    return (r @ p).tolist()


def generate_test_poses(
    target_position, target_orientation_xyzw,
    horiz=0, vert=0, distance=350, roll=-90, pitch=15, yaws=(-30, 0, 30),
):
    """Generate world-frame poses for the test."""
    poses = []
    pivot_local = (_mm_to_m(horiz), 0.0, _mm_to_m(vert))

    for yaw in yaws:
        offset = _spherical_to_cartesian(pitch, yaw, distance)
        pos_local = tuple(pivot_local[i] + offset[i] for i in range(3))
        q_local = _look_at_quaternion(pos_local, pivot_local, roll)

        # Transform to base_link frame
        pos_world_rot = _quat_rot_point(target_orientation_xyzw, pos_local)
        pos_world = [
            target_position[i] + pos_world_rot[i] for i in range(3)
        ]
        q_world = _quat_multiply(target_orientation_xyzw, q_local)
        q_world = _normalize_quat(q_world)

        name = f"H{horiz}_V{vert}_D{distance}_R{roll}_P{pitch}_Y{yaw}"
        poses.append({
            "name": name,
            "position": pos_world,
            "orientation": list(q_world),  # [qx, qy, qz, qw]
            "parameters": {
                "horiz": horiz, "vert": vert, "distance": distance,
                "roll": roll, "pitch": pitch, "yaw": yaw,
            },
        })

    return poses


# ── WebSocket helpers ────────────────────────────────────────────


class GatewayTestClient:
    def __init__(self, server: str, port: int = 8766):
        self.uri = f"ws://{server}:{port}"
        self.ws: Optional[Any] = None
        self._req_id = 0
        self._pending: Dict[str, asyncio.Future] = {}
        self._feedback: List[Dict] = []
        self._listen_task: Optional[asyncio.Task] = None

    async def connect(self, timeout=10.0):
        print(f"Connecting to {self.uri}…")
        self.ws = await asyncio.wait_for(
            websockets.connect(self.uri, ping_interval=30, ping_timeout=300),
            timeout=timeout,
        )
        self._listen_task = asyncio.create_task(self._listener())
        print("  ✓ Connected")

    async def close(self):
        if self._listen_task:
            self._listen_task.cancel()
        if self.ws:
            await self.ws.close()

    async def _listener(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                t = msg.get("type", "")
                rid = msg.get("request_id")

                if t == "rpc_feedback":
                    self._feedback.append(msg)
                    idx = msg.get("current_pose_index", "?")
                    total = msg.get("total_poses", "?")
                    status = msg.get("status", "")
                    pname = msg.get("current_pose_name", "")
                    pct = msg.get("progress_percent", 0)
                    print(
                        f"  ⟳ [{idx+1 if isinstance(idx,int) else idx}/{total}] "
                        f"{status} {pname} ({pct:.0f}%)"
                    )
                elif rid and rid in self._pending:
                    fut = self._pending.pop(rid)
                    if not fut.done():
                        fut.set_result(msg)
                elif t == "heartbeat_ack":
                    pass
                elif t == "pong":
                    pass
                else:
                    print(f"  ← {t}: {json.dumps(msg)[:200]}")
        except websockets.ConnectionClosed:
            pass

    def _next_rid(self) -> str:
        self._req_id += 1
        return f"test_{self._req_id}"

    async def rpc(
        self, method: str, params: dict = None, timeout: float = 300.0
    ) -> dict:
        rid = self._next_rid()
        req = {
            "type": "rpc",
            "request_id": rid,
            "method": method,
            "params": params or {},
        }
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self.ws.send(json.dumps(req))
        result = await asyncio.wait_for(fut, timeout=timeout)
        return result

    async def ping(self) -> float:
        rid = self._next_rid()
        req = {"type": "ping", "request_id": rid, "timestamp_ns": time.time_ns()}
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self.ws.send(json.dumps(req))
        t0 = time.time()
        await asyncio.wait_for(fut, timeout=5.0)
        return (time.time() - t0) * 1000


# ── Test runner ──────────────────────────────────────────────────


async def run_test(args):
    # Determine mode
    if args.real_only:
        test_mode = "real"
    else:
        test_mode = "simulation"

    print(f"═══════════════════════════════════════")
    print(f"  cuRobo Gateway Test — {test_mode.upper()} mode")
    if test_mode == "real":
        print(f"  ⚠  Real robot will move!  Speed={args.speed}")
        print(f"  ⚠  Uses RTDE servoJ (bypasses ROS2 control stack)")
    print(f"═══════════════════════════════════════\n")

    client = GatewayTestClient(args.server, args.port)
    await client.connect()

    # Ping
    rtt = await client.ping()
    print(f"  Ping: {rtt:.1f} ms")

    # ── Environment info ─────────────────────────────────────────
    print("\n═══ Environment Info ═══")
    env = await client.rpc("get_environment_info")
    if not env.get("success"):
        print(f"  ✗ Failed: {env}")
        await client.close()
        return

    robots = env.get("robots", [])
    objects = env.get("objects", [])
    obj_tf = env.get("object_transforms", {})
    print(f"  Robots:  {robots}")
    print(f"  Objects: {objects}")

    face_tf = obj_tf.get("face_link")
    if face_tf:
        print(f"  face_link pos: {face_tf['position']}")
        print(f"  face_link ori: {face_tf['orientation']}")
    else:
        print("  ⚠ face_link TF not available — using default")
        face_tf = {
            "position": [0.1742, 0.0, 0.57],
            "orientation": [0.0, 0.0, 0.7071, 0.7071],
        }

    # ── Robot status ─────────────────────────────────────────────
    robot_name = args.robot or (robots[0] if robots else "nakul_ur5e")
    print(f"\n═══ Robot Status ({robot_name}) ═══")
    status = await client.rpc("get_robot_status", {"robot_name": robot_name})
    for k, v in status.items():
        if k not in ("type", "request_id"):
            print(f"  {k}: {v}")

    # ── Generate poses ───────────────────────────────────────────
    print("\n═══ Generating Test Poses ═══")
    poses = generate_test_poses(
        target_position=face_tf["position"],
        target_orientation_xyzw=face_tf["orientation"],
        horiz=0, vert=0, distance=350,
        roll=-90, pitch=15, yaws=[-30, 0, 30],
    )
    print(f"  Generated {len(poses)} poses:")
    for p in poses:
        print(f"    {p['name']}: pos={[f'{x:.4f}' for x in p['position']]}")

    # ── Simulation run ───────────────────────────────────────────
    if test_mode == "simulation":
        print("\n═══ Simulation Mode ═══")
        prep = await client.rpc("prepare_mode", {"mode": "simulation"}, timeout=120)
        print(f"  Prepare: {prep.get('message')}")

        if prep.get("ready") or prep.get("success"):
            # Go home first
            print(f"  Moving {robot_name} to HOME…")
            home_result = await client.rpc(
                "move_home", {"robot_name": robot_name}, timeout=60,
            )
            print(f"  Home: {home_result.get('message')}")

            # Run poses
            print(f"  Running proto_sim (simulation)…")
            client._feedback.clear()
            result = await client.rpc(
                "run_proto_sim",
                {
                    "robot_name": robot_name,
                    "poses": poses,
                    "idle_time": 1.0,
                    "mode": "simulation",
                    "move_speed": args.speed,
                },
                timeout=300,
            )
            print(f"\n  Result: {result.get('message')}")
            print(f"    completed={result.get('completed')}, "
                  f"ik_failed={result.get('ik_failed')}, "
                  f"real_failed={result.get('real_failed')}")

            # Go home after
            print(f"  Returning {robot_name} to HOME…")
            home_result = await client.rpc(
                "move_home", {"robot_name": robot_name}, timeout=60,
            )
            print(f"  Home: {home_result.get('message')}")
        else:
            print(f"  ✗ Mode not ready: {prep.get('message')}")

    # ── Real robot run ───────────────────────────────────────────
    if test_mode == "real":
        print("\n═══ Real Robot Mode (RTDE) ═══")
        print(f"  Speed: {args.speed}")
        prep = await client.rpc("prepare_mode", {"mode": "real"}, timeout=180)
        print(f"  Prepare: {prep.get('message')}")

        if prep.get("ready"):
            # Re-check robot status after mode switch
            status2 = await client.rpc("get_robot_status", {"robot_name": robot_name})
            print(f"  Robot status: mode={status2.get('current_mode')}, "
                  f"real_available={status2.get('real_robot_available')}")
            conn = status2.get("connection_details", {})
            print(f"  RTDE: {conn.get('rtde', 'unknown')}")
            print(f"  cuRobo: {conn.get('curobo', 'unknown')}")

            # Go home first
            print(f"\n  Moving {robot_name} to HOME…")
            home_result = await client.rpc(
                "move_home", {"robot_name": robot_name}, timeout=60,
            )
            print(f"  Home: {home_result.get('message')}")

            # Run poses
            print(f"  Running proto_sim (real) with RTDE servoJ…")
            client._feedback.clear()
            result = await client.rpc(
                "run_proto_sim",
                {
                    "robot_name": robot_name,
                    "poses": poses,
                    "idle_time": 2.0,
                    "mode": "real",
                    "move_speed": args.speed,
                },
                timeout=300,
            )
            print(f"\n  Result: {result.get('message')}")
            print(f"    completed={result.get('completed')}, "
                  f"ik_failed={result.get('ik_failed')}, "
                  f"real_failed={result.get('real_failed')}")
            if result.get("hardware_issues"):
                print(f"    ⚠ hardware_issues: {result['hardware_issues']}")

            # Go home after
            print(f"  Returning {robot_name} to HOME…")
            home_result = await client.rpc(
                "move_home", {"robot_name": robot_name}, timeout=60,
            )
            print(f"  Home: {home_result.get('message')}")
        else:
            print(f"  ✗ Mode not ready: {prep.get('message')}")

    # ── Cleanup ──────────────────────────────────────────────────
    print("\n═══ Test Complete ═══")
    await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test NVIDIA cuRobo gateway (sim or real mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Simulation test (default)
  python test_curobo_gateway.py --server 10.0.0.5

  # Real robot test at conservative speed
  python test_curobo_gateway.py --server 10.0.0.5 --real-only --speed 0.15

  # Real robot test at moderate speed (default)
  python test_curobo_gateway.py --server 10.0.0.5 --real-only --speed 0.3

  # Real robot test at higher speed
  python test_curobo_gateway.py --server 10.0.0.5 --real-only --speed 0.5
""",
    )
    parser.add_argument(
        "--server", default="localhost",
        help="Server IP address (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8766,
        help="Gateway WebSocket port (default: 8766)",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sim-only", action="store_true", default=True,
        help="Run simulation only (default)",
    )
    mode_group.add_argument(
        "--real-only", action="store_true",
        help="Run on real robot only (uses RTDE servoJ, no ROS2 control stack)",
    )

    parser.add_argument(
        "--speed", type=float, default=0.3,
        help="Velocity scaling factor 0.01–1.0 (default: 0.3). "
             "Maps to cuRobo time_dilation_factor (0.3 = 30%% max speed). "
             "Lower = slower = safer for real robot.",
    )
    parser.add_argument(
        "--robot", type=str, default=None,
        help="Robot name to test (default: first robot in environment)",
    )
    args = parser.parse_args()

    # Clamp speed
    args.speed = max(0.01, min(args.speed, 1.0))

    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
