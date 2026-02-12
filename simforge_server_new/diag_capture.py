#!/usr/bin/env python3
"""
Trajectory Diagnostic: Send a known trajectory and record what the controller
actually does (commanded vs actual positions, velocities, timing jitter).

This script:
1. Reads the current joint state
2. Creates a simple 2-point trajectory (small move on joint 1)
3. Sends it to the ScaledJointTrajectoryController
4. Records the controller_state topic at ~100Hz during execution
5. Analyzes timing, velocity, and jitter

Run inside the container:
  python3 /ros2_ws/diag_capture.py
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
import math
import threading


class TrajectoryDiag(Node):
    def __init__(self):
        super().__init__("traj_diag")

        self.joint_names = [
            "nakul_shoulder_pan_joint",
            "nakul_shoulder_lift_joint",
            "nakul_elbow_joint",
            "nakul_wrist_1_joint",
            "nakul_wrist_2_joint",
            "nakul_wrist_3_joint",
        ]

        self.current_positions = None
        self.controller_samples = []  # (timestamp, desired_pos, actual_pos, error, desired_vel, actual_vel)
        self.collecting = False
        self.traj_sent_points = []  # the actual JointTrajectoryPoints we sent

        # Subscribe to joint states to get current position
        self.js_sub = self.create_subscription(
            JointState, "/joint_states", self._js_cb, 10
        )

        # Subscribe to controller state for diagnostics
        self.ctrl_sub = self.create_subscription(
            JointTrajectoryControllerState,
            "/nakul_scaled_joint_trajectory_controller/controller_state",
            self._ctrl_cb,
            100,
        )

        # Action client
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/nakul_scaled_joint_trajectory_controller/follow_joint_trajectory",
        )

        self.get_logger().info("Waiting for current joint state...")

    def _js_cb(self, msg):
        if self.current_positions is not None:
            return
        # Extract nakul joints in order
        positions = {}
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_names:
                positions[name] = pos
        if len(positions) == 6:
            self.current_positions = [positions[n] for n in self.joint_names]
            self.get_logger().info(
                f"Current joints: {[f'{p:.4f}' for p in self.current_positions]}"
            )

    def _ctrl_cb(self, msg):
        if not self.collecting:
            return

        t = self.get_clock().now().nanoseconds * 1e-9
        sample = {
            "time": t,
            "desired_pos": list(msg.desired.positions) if msg.desired.positions else [],
            "actual_pos": list(msg.actual.positions) if msg.actual.positions else [],
            "error_pos": list(msg.error.positions) if msg.error.positions else [],
            "desired_vel": list(msg.desired.velocities) if msg.desired.velocities else [],
            "actual_vel": list(msg.actual.velocities) if msg.actual.velocities else [],
        }
        self.controller_samples.append(sample)

    def send_test_trajectory(self):
        """Send a small test trajectory and capture diagnostics."""
        if self.current_positions is None:
            self.get_logger().error("No joint state received!")
            return False

        self.get_logger().info("Waiting for action server...")
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available!")
            return False

        # Create a trajectory: current -> small move on shoulder_pan -> back
        start = list(self.current_positions)
        mid = list(start)
        mid[0] += 0.3  # Move shoulder_pan by 0.3 rad (~17 degrees)

        # Build trajectory with the SAME retiming approach as command_gateway
        # resample_dt=0.02 means TOTG gives us points every 20ms
        # At 0.17 rad/s safe_vel and 0.3 rad movement:
        # time = 0.3 / 0.17 = ~1.76 seconds

        # But let's test TWO different trajectory styles:
        # Test 1: Dense trajectory (like TOTG output) - many points with velocities
        # Test 2: Sparse trajectory (just start+end) - let controller interpolate

        self.get_logger().info("=" * 60)
        self.get_logger().info("TEST 1: Dense trajectory (simulating TOTG + retiming)")
        self.get_logger().info("=" * 60)
        ok1 = self._run_trajectory(start, mid, dense=True, label="dense")

        if not ok1:
            self.get_logger().warn("Dense trajectory failed, skipping sparse test")
            self._analyze()
            return False

        # Wait 3 seconds between tests
        time.sleep(3.0)

        self.get_logger().info("=" * 60)
        self.get_logger().info("TEST 2: Sparse trajectory (2 points, no velocities)")
        self.get_logger().info("=" * 60)
        ok2 = self._run_trajectory(mid, start, dense=False, label="sparse")

        self._analyze()
        return True

    def _run_trajectory(self, start_pos, end_pos, dense=True, label=""):
        """Build and send a trajectory, return True if succeeded."""
        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)

        move_amount = max(abs(end_pos[j] - start_pos[j]) for j in range(6))
        safe_vel = 0.17  # matches our retiming: 0.2 * 0.85

        if dense:
            # Generate points every 0.02s (like resample_dt)
            total_time = move_amount / safe_vel
            dt = 0.02
            n_points = max(int(total_time / dt), 2)

            for i in range(n_points + 1):
                frac = i / n_points
                t = frac * total_time

                pt = JointTrajectoryPoint()
                pt.positions = [
                    start_pos[j] + frac * (end_pos[j] - start_pos[j])
                    for j in range(6)
                ]

                # Velocities: zero at start/end, constant in middle
                if i == 0 or i == n_points:
                    pt.velocities = [0.0] * 6
                else:
                    pt.velocities = [
                        (end_pos[j] - start_pos[j]) / total_time
                        for j in range(6)
                    ]

                # NO accelerations (cubic Hermite only)
                pt.accelerations = []

                sec = int(t)
                nsec = int((t - sec) * 1e9)
                pt.time_from_start = Duration(sec=sec, nanosec=nsec)
                traj.points.append(pt)

            self.get_logger().info(
                f"[{label}] Sending {len(traj.points)} points, "
                f"duration={total_time:.2f}s, safe_vel={safe_vel:.3f}"
            )
        else:
            # Just 2 points: start and end
            total_time = move_amount / safe_vel

            pt0 = JointTrajectoryPoint()
            pt0.positions = list(start_pos)
            pt0.velocities = [0.0] * 6
            pt0.time_from_start = Duration(sec=0, nanosec=0)
            traj.points.append(pt0)

            pt1 = JointTrajectoryPoint()
            pt1.positions = list(end_pos)
            pt1.velocities = [0.0] * 6
            sec = int(total_time)
            nsec = int((total_time - sec) * 1e9)
            pt1.time_from_start = Duration(sec=sec, nanosec=nsec)
            traj.points.append(pt1)

            self.get_logger().info(
                f"[{label}] Sending 2 points, duration={total_time:.2f}s"
            )

        # Save for analysis
        self.traj_sent_points.append({
            "label": label,
            "points": [(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9,
                         list(p.positions), list(p.velocities))
                        for p in traj.points],
        })

        # Send via action
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.collecting = True
        collect_start = len(self.controller_samples)

        future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"[{label}] Goal rejected!")
            self.collecting = False
            return False

        self.get_logger().info(f"[{label}] Goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)

        self.collecting = False
        result = result_future.result()
        if result is None:
            self.get_logger().error(f"[{label}] No result received!")
            return False

        status = result.result.error_code
        self.get_logger().info(f"[{label}] Result: error_code={status}")

        n_samples = len(self.controller_samples) - collect_start
        self.get_logger().info(f"[{label}] Captured {n_samples} controller_state samples")
        return True

    def _analyze(self):
        """Analyze captured controller_state samples."""
        if not self.controller_samples:
            self.get_logger().warn("No controller_state samples captured!")
            return

        self.get_logger().info("")
        self.get_logger().info("=" * 70)
        self.get_logger().info("ANALYSIS")
        self.get_logger().info("=" * 70)

        samples = self.controller_samples
        n = len(samples)
        self.get_logger().info(f"Total samples: {n}")

        if n < 2:
            return

        # Timing analysis
        dts = []
        for i in range(1, n):
            dt = samples[i]["time"] - samples[i - 1]["time"]
            dts.append(dt)

        avg_dt = sum(dts) / len(dts)
        min_dt = min(dts)
        max_dt = max(dts)
        jitter_dts = [abs(dt - avg_dt) for dt in dts]
        max_jitter = max(jitter_dts)
        p95_jitter = sorted(jitter_dts)[int(0.95 * len(jitter_dts))]

        self.get_logger().info(f"\n--- Timing (controller_state publish rate) ---")
        self.get_logger().info(f"  avg dt: {avg_dt*1000:.1f} ms  (expected ~10ms at 100Hz)")
        self.get_logger().info(f"  min dt: {min_dt*1000:.1f} ms")
        self.get_logger().info(f"  max dt: {max_dt*1000:.1f} ms")
        self.get_logger().info(f"  max jitter: {max_jitter*1000:.1f} ms")
        self.get_logger().info(f"  p95 jitter: {p95_jitter*1000:.1f} ms")

        # Count timing outliers (>2x avg)
        outliers = sum(1 for dt in dts if dt > 2 * avg_dt)
        self.get_logger().info(f"  timing outliers (>2x avg): {outliers}/{len(dts)}")

        # Velocity analysis - actual joint velocities
        self.get_logger().info(f"\n--- Joint Velocities (from actual positions) ---")
        for j in range(6):
            vels = []
            for i in range(1, n):
                dt = samples[i]["time"] - samples[i - 1]["time"]
                if dt > 0 and samples[i]["actual_pos"] and samples[i - 1]["actual_pos"]:
                    v = abs(
                        samples[i]["actual_pos"][j] - samples[i - 1]["actual_pos"][j]
                    ) / dt
                    vels.append(v)
            if vels:
                self.get_logger().info(
                    f"  Joint {j}: max={max(vels):.4f} rad/s, "
                    f"avg={sum(vels)/len(vels):.4f} rad/s, "
                    f"p95={sorted(vels)[int(0.95*len(vels))]:.4f} rad/s"
                )

        # Desired velocity analysis
        self.get_logger().info(f"\n--- Desired Velocities (from controller) ---")
        for j in range(6):
            dvels = []
            for s in samples:
                if s["desired_vel"] and len(s["desired_vel"]) > j:
                    dvels.append(abs(s["desired_vel"][j]))
            if dvels:
                self.get_logger().info(
                    f"  Joint {j}: max={max(dvels):.4f} rad/s, "
                    f"avg={sum(dvels)/len(dvels):.4f} rad/s"
                )

        # Tracking error analysis
        self.get_logger().info(f"\n--- Tracking Error (desired - actual) ---")
        for j in range(6):
            errs = []
            for s in samples:
                if s["error_pos"] and len(s["error_pos"]) > j:
                    errs.append(abs(s["error_pos"][j]))
            if errs:
                self.get_logger().info(
                    f"  Joint {j}: max={max(errs):.6f} rad, "
                    f"avg={sum(errs)/len(errs):.6f} rad"
                )

        # Velocity jumps (jerk) analysis - key diagnostic
        self.get_logger().info(f"\n--- Velocity Jumps (indicates jerky motion) ---")
        for j in range(6):
            vel_jumps = []
            prev_vel = None
            for i in range(1, n):
                dt = samples[i]["time"] - samples[i - 1]["time"]
                if dt > 0 and samples[i]["actual_pos"] and samples[i - 1]["actual_pos"]:
                    v = (samples[i]["actual_pos"][j] - samples[i - 1]["actual_pos"][j]) / dt
                    if prev_vel is not None:
                        jump = abs(v - prev_vel)
                        vel_jumps.append(jump)
                    prev_vel = v
            if vel_jumps:
                max_jump = max(vel_jumps)
                p95_jump = sorted(vel_jumps)[int(0.95 * len(vel_jumps))]
                p99_jump = sorted(vel_jumps)[int(0.99 * len(vel_jumps))]
                self.get_logger().info(
                    f"  Joint {j}: max_jump={max_jump:.4f} rad/s, "
                    f"p95={p95_jump:.4f}, p99={p99_jump:.4f} rad/s"
                )

        # Desired velocity jumps (what the controller COMMANDS)
        self.get_logger().info(f"\n--- Desired Velocity Jumps (controller interpolation) ---")
        for j in range(6):
            dvel_jumps = []
            for i in range(1, n):
                if (samples[i]["desired_vel"] and samples[i-1]["desired_vel"]
                    and len(samples[i]["desired_vel"]) > j
                    and len(samples[i-1]["desired_vel"]) > j):
                    jump = abs(samples[i]["desired_vel"][j] - samples[i-1]["desired_vel"][j])
                    dvel_jumps.append(jump)
            if dvel_jumps:
                max_jump = max(dvel_jumps)
                p95_jump = sorted(dvel_jumps)[int(0.95 * len(dvel_jumps))]
                self.get_logger().info(
                    f"  Joint {j}: max_jump={max_jump:.4f} rad/s, "
                    f"p95={p95_jump:.4f} rad/s"
                )

        # Print first 20 and last 20 samples for visual inspection
        self.get_logger().info(f"\n--- First 20 samples (desired_vel joint 0) ---")
        for i, s in enumerate(samples[:20]):
            t = s["time"] - samples[0]["time"]
            dv = s["desired_vel"][0] if s["desired_vel"] else 0
            av = s["actual_vel"][0] if s["actual_vel"] else 0
            dp = s["desired_pos"][0] if s["desired_pos"] else 0
            ap = s["actual_pos"][0] if s["actual_pos"] else 0
            err = s["error_pos"][0] if s["error_pos"] else 0
            self.get_logger().info(
                f"  t={t:.3f}s  d_pos={dp:.5f}  a_pos={ap:.5f}  "
                f"err={err:.6f}  d_vel={dv:.5f}  a_vel={av:.5f}"
            )


def main():
    rclpy.init()
    node = TrajectoryDiag()

    # Spin until we get joint state
    timeout = time.time() + 10.0
    while node.current_positions is None and time.time() < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)

    if node.current_positions is None:
        node.get_logger().error("Timeout waiting for joint state!")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Run the test
    node.send_test_trajectory()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
