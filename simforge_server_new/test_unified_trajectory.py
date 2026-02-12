#!/usr/bin/env python3
"""
ALL-IN-ONE safe test: restart container → switch to real → run trajectory.

Valid8 Dual Cell environment:
  - Optical table surface at z=1.0 in world  (bz = -0.03 in base_link)
  - nakul_ur5e base at (-0.6758, 0, 1.03) with yaw=-90°
  - UR5e home [0,-π/2,0,-π/2,0,0] → tool_tip at ~(-0.080, 0.283, 1.079) base_link
  - tool_tip_link orientation at home ≈ (0, 0, 0, 1) — tool pointing UP in base_link Z
  - Face at ~(0.0, 0.85, 0.57) in nakul_base_link

IMPORTANT: The home FK position z=1.079 is at the very edge of the UR5e
workspace.  ANY lateral movement at that height is IK-unreachable!
Test poses must be at a lower Z so the arm has room to manoeuvre.

SAFETY RULES:
  1. move_speed = 0.08 (8% velocity — moderate, safe)
  2. Only 3 gentle poses — small translations in a reachable region
  3. All poses well above table (z≥0.85 in base_link, table at z=-0.03)
  4. Orientation (0,0,0,1) = tool pointing UP in base_link Z

Usage:
    python3 test_unified_trajectory.py
"""

import asyncio
import json
import subprocess
import time
import sys

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)


WS_URL = "ws://localhost:8766"
COMPOSE_DIR = "/home/badal/simforge/simforge_server_new/docker"

# ── SAFE test poses in nakul_base_link frame ────────────────────────
# Home FK: tool_tip at (-0.080, 0.283, 1.079) — but z=1.079 is at the
# very edge of the UR5e workspace. Any lateral move at that height is
# IK-unreachable!  We drop 15 cm to z=0.93 where the arm has full
# dexterity, then do small (3–5 cm) lateral translations.
# All poses are well above the table (z=0.93 base_link → z≈1.96 world).
MOVE_SPEED = 0.08  # 8% velocity — moderate and safe

TEST_POSES = [
    {
        "name": "start_pos",
        "position": [-0.08, 0.35, 0.93],     # slightly forward + lower from home
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    {
        "name": "nudge_right",
        "position": [-0.03, 0.35, 0.93],     # +5cm in X (right)
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    {
        "name": "nudge_left",
        "position": [-0.13, 0.35, 0.93],     # -5cm in X (left)
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
]


def run_cmd(cmd, cwd=COMPOSE_DIR, check=True):
    """Run a shell command and print output."""
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n")[-5:]:
            print(f"    {line}")
    if result.returncode != 0 and check:
        print(f"    STDERR: {result.stderr.strip()[-200:]}")
    return result


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Restart Docker container
# ═══════════════════════════════════════════════════════════════════
def step1_restart_container():
    print("\n" + "=" * 60)
    print("STEP 1: Restarting Docker container")
    print("=" * 60)
    run_cmd("sudo docker compose --profile sim down")
    run_cmd("sudo docker compose --profile sim up -d")
    print("  Waiting 20s for ROS2 stack to initialize...")
    time.sleep(20)

    # Wait for WebSocket to be available
    for attempt in range(30):
        try:
            result = subprocess.run(
                ["sudo", "docker", "logs", "simforge-server-sim"],
                capture_output=True, text=True,
            )
            if "WebSocket server started" in result.stdout:
                print("  ✓ WebSocket server is up")
                return True
        except Exception:
            pass
        time.sleep(2)

    print("  ✗ WebSocket server did not start in time")
    return False


# ═══════════════════════════════════════════════════════════════════
# STEP 2: Switch to real robot mode via RPC
# ═══════════════════════════════════════════════════════════════════
async def step2_switch_to_real(ws):
    print("\n" + "=" * 60)
    print("STEP 2: Switching to REAL robot mode")
    print("=" * 60)

    rpc = {
        "type": "rpc",
        "request_id": "mode_switch_001",
        "method": "prepare_mode",
        "params": {"mode": "both"},
    }
    await ws.send(json.dumps(rpc))
    print("  Sent prepare_mode(both) — waiting for mode switch...")

    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=120)
            msg = json.loads(raw)

            if msg.get("type") == "heartbeat":
                continue

            if msg.get("type") == "rpc_result":
                success = msg.get("success", False)
                ready = msg.get("ready", False)
                message = msg.get("message", "")
                print(f"  Result: success={success}, ready={ready}")
                print(f"  Message: {message}")

                if success and ready:
                    print("  ✓ Real robot mode activated")
                    # Wait a moment for controllers to fully settle
                    await asyncio.sleep(3)
                    return True
                else:
                    print("  ✗ Mode switch failed!")
                    return False

            elif msg.get("type") == "rpc_feedback":
                print(f"  ... {msg.get('message', msg.get('status', ''))}")

        except asyncio.TimeoutError:
            print("  ✗ Timeout waiting for mode switch!")
            return False


# ═══════════════════════════════════════════════════════════════════
# STEP 3: Run the trajectory test
# ═══════════════════════════════════════════════════════════════════
async def step3_run_trajectory(ws):
    print("\n" + "=" * 60)
    print("STEP 3: Running trajectory test")
    print("=" * 60)
    print(f"  Poses: {len(TEST_POSES)}")
    for p in TEST_POSES:
        print(f"    {p['name']}: pos={p['position']}")
    print(f"  Speed: {MOVE_SPEED*100:.0f}% ({MOVE_SPEED})")
    print()

    rpc = {
        "type": "rpc",
        "request_id": "test_safe_001",
        "method": "run_proto_sim",
        "params": {
            "robot_name": "nakul_ur5e",
            "poses": TEST_POSES,
            "idle_time": 0.0,
            "mode": "both",
            "move_speed": MOVE_SPEED,
        },
    }
    await ws.send(json.dumps(rpc))

    feedback_msgs = []
    result_msg = None
    t_start = time.monotonic()

    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=300)
            msg = json.loads(raw)
            elapsed = time.monotonic() - t_start
            msg_type = msg.get("type", "?")

            if msg_type == "rpc_feedback":
                status = msg.get("status", "?")
                pose_idx = msg.get("current_pose_index", "?")
                pose_name = msg.get("current_pose_name", "?")
                pct = msg.get("progress_percent", 0)
                message = msg.get("message", "")
                print(
                    f"  [{elapsed:6.1f}s] {status:12s}  "
                    f"pose={pose_idx}/{len(TEST_POSES)}  "
                    f"name={pose_name:18s}  {pct:.0f}%"
                    + (f"  {message}" if message else "")
                )
                feedback_msgs.append(msg)

            elif msg_type == "rpc_result":
                result_msg = msg
                print(f"\n  [{elapsed:6.1f}s] RESULT:")
                print(f"    success:   {msg.get('success')}")
                print(f"    completed: {msg.get('completed')}/{msg.get('total')}")
                print(f"    message:   {msg.get('message')}")
                break

            elif msg_type == "heartbeat":
                pass

        except asyncio.TimeoutError:
            print("\n  Timeout!")
            break

    # ── Analysis ────────────────────────────────────────────────
    planning = [m for m in feedback_msgs if m.get("status") == "planning"]
    reached = [m for m in feedback_msgs if m.get("status") == "reached"]
    moving = [m for m in feedback_msgs if m.get("status") == "moving"]

    print("\n  ANALYSIS:")
    if planning and not moving:
        print("  ✓ UNIFIED multi-waypoint execution")
    elif moving:
        print("  ✗ LEGACY pose-by-pose (fallback)")
    print(f"  Wall time: {time.monotonic() - t_start:.1f}s")

    return result_msg and result_msg.get("success", False)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
async def main():
    print("=" * 60)
    print("ALL-IN-ONE TEST: restart → real mode → trajectory")
    print("=" * 60)

    # Step 1: Restart container
    if not step1_restart_container():
        print("\nABORTED: Container failed to start")
        sys.exit(1)

    # Connect WebSocket
    print(f"\n  Connecting to {WS_URL}...")
    try:
        ws = await websockets.connect(
            WS_URL, ping_interval=30, ping_timeout=300,
        )
    except Exception as e:
        print(f"\n  Failed to connect: {e}")
        sys.exit(1)
    print("  ✓ Connected")

    try:
        # Step 2: Switch to real robot mode
        if not await step2_switch_to_real(ws):
            print("\nABORTED: Mode switch failed")
            sys.exit(1)

        # Step 3: Run trajectory
        ok = await step3_run_trajectory(ws)

        print("\n" + "=" * 60)
        if ok:
            print("TEST PASSED ✓")
        else:
            print("TEST FAILED ✗")
        print("=" * 60)

    finally:
        await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
