#!/usr/bin/env python3
"""
Position Accuracy Diagnostic — measures how accurately the UR robot
reaches planned final waypoints after servoJ streaming completes.

Run inside the Docker container:
  python3 /workspace/simforge_server_nvidia/tools/position_accuracy_diag.py

Connects to the running gateway WebSocket (port 8766) and also
directly to the UR20 RTDE interface to measure actual vs planned positions.
"""

import json
import math
import sys
import time

try:
    import rtde_receive
except ImportError:
    print("ERROR: ur_rtde not installed — run inside container")
    sys.exit(1)


UR20_IP = "192.168.0.40"


def rad2deg(r):
    return r * 180.0 / math.pi


def main():
    print(f"Connecting RTDE receive to {UR20_IP}...")
    recv = rtde_receive.RTDEReceiveInterface(UR20_IP)
    print("Connected ✓")

    # Read current position
    q = list(recv.getActualQ())
    print(f"\nCurrent joint positions (rad): "
          f"[{', '.join(f'{v:.5f}' for v in q)}]")
    print(f"Current joint positions (deg): "
          f"[{', '.join(f'{rad2deg(v):.2f}' for v in q)}]")

    # Read safety/robot mode
    safety = recv.getSafetyMode()
    robot_mode = recv.getRobotMode()
    print(f"Safety mode: {safety}, Robot mode: {robot_mode}")

    # Sit and sample position stability for 3 seconds
    print("\n--- Position stability test (3s, 20Hz) ---")
    samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        q_now = list(recv.getActualQ())
        samples.append(q_now)
        time.sleep(0.05)

    # Compute max deviation from mean for each joint
    n = len(samples)
    means = [sum(s[j] for s in samples) / n for j in range(6)]
    max_devs = [
        max(abs(s[j] - means[j]) for s in samples) for j in range(6)
    ]
    print(f"Samples: {n}")
    print(f"Mean:     [{', '.join(f'{v:.5f}' for v in means)}]")
    print(f"Max dev:  [{', '.join(f'{rad2deg(v)*1000:.2f}' for v in max_devs)}] mDeg")
    total_dev_deg = math.sqrt(sum(d**2 for d in max_devs)) * 180.0 / math.pi
    print(f"RSS max deviation: {total_dev_deg*1000:.2f} mDeg")

    # Now let's measure what happens after a servoJ trajectory
    # by doing a small test move with the control interface.
    print("\n--- servoJ accuracy test ---")
    print("This will perform a SMALL test move (~2 degrees) and measure error.")
    
    input_str = input("Press Enter to continue or 'q' to skip: ").strip()
    if input_str.lower() == 'q':
        print("Skipped.")
        recv.disconnect()
        return

    import rtde_control

    q_start = list(recv.getActualQ())
    print(f"Start position: [{', '.join(f'{v:.5f}' for v in q_start)}]")

    # Create a trajectory: move joint 0 by +2 degrees and back
    delta = math.radians(2.0)
    q_mid = list(q_start)
    q_mid[0] += delta

    # Simple trajectory: 50 waypoints out, 50 back
    n_pts = 50
    out_traj = []
    for i in range(n_pts + 1):
        alpha = i / n_pts
        q = [q_start[j] + alpha * (q_mid[j] - q_start[j]) for j in range(6)]
        out_traj.append(q)
    back_traj = []
    for i in range(n_pts + 1):
        alpha = i / n_pts
        q = [q_mid[j] + alpha * (q_start[j] - q_mid[j]) for j in range(6)]
        back_traj.append(q)

    full_traj = out_traj + back_traj[1:]  # avoid duplicate mid point
    target_final = full_traj[-1]

    dt = 0.02  # 50 Hz
    total_time = len(full_traj) * dt

    print(f"Trajectory: {len(full_traj)} pts, {total_time:.1f}s, "
          f"J0 moves ±{2.0:.1f}° and returns")
    print(f"Target final: [{', '.join(f'{v:.5f}' for v in target_final)}]")

    # Disconnect recv before servoJ (it will die anyway)
    recv.disconnect()

    ctrl = rtde_control.RTDEControlInterface(UR20_IP)
    print("Control interface connected ✓")

    # Stream with servoJ
    for i, q_target in enumerate(full_traj):
        t_cmd = time.monotonic()
        ctrl.servoJ(q_target, 0.0, 0.0, dt, 0.1, 300)
        elapsed = time.monotonic() - t_cmd
        remaining = dt - elapsed
        if remaining > 0.001:
            time.sleep(remaining)

    ctrl.servoStop()
    time.sleep(0.5)  # let the robot settle
    ctrl.disconnect()

    # Reconnect recv and read actual position
    recv2 = rtde_receive.RTDEReceiveInterface(UR20_IP)
    time.sleep(0.2)  # settle

    # Take multiple samples
    post_samples = []
    for _ in range(20):
        post_samples.append(list(recv2.getActualQ()))
        time.sleep(0.05)

    q_actual = [sum(s[j] for s in post_samples) / len(post_samples) for j in range(6)]

    errors_rad = [q_actual[j] - target_final[j] for j in range(6)]
    errors_deg = [rad2deg(e) for e in errors_rad]
    rss_error_deg = math.sqrt(sum(e**2 for e in errors_deg))

    print(f"\n--- Results ---")
    print(f"Target final:  [{', '.join(f'{v:.5f}' for v in target_final)}]")
    print(f"Actual final:  [{', '.join(f'{v:.5f}' for v in q_actual)}]")
    print(f"Error (deg):   [{', '.join(f'{e:.4f}' for e in errors_deg)}]")
    print(f"Error (mDeg):  [{', '.join(f'{e*1000:.1f}' for e in errors_deg)}]")
    print(f"RSS error:     {rss_error_deg:.4f}° ({rss_error_deg*1000:.1f} mDeg)")

    # The key metric: this is what causes the jump
    if rss_error_deg > 0.1:
        print(f"\n⚠ SIGNIFICANT ERROR: {rss_error_deg:.2f}° — this explains the jumps!")
    elif rss_error_deg > 0.01:
        print(f"\n⚠ MODERATE ERROR: {rss_error_deg*1000:.0f} mDeg — visible at high speed")
    else:
        print(f"\n✓ EXCELLENT ACCURACY: {rss_error_deg*1000:.0f} mDeg")

    recv2.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
