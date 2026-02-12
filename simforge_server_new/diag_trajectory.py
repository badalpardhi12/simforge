#!/usr/bin/env python3
"""
Diagnostic: Dump a planned trajectory's waypoints (positions, velocities, timestamps)
so we can see exactly what the ScaledJointTrajectoryController receives.

Run inside the container:
  python3 /ros2_ws/diag_trajectory.py

This subscribes to the trajectory topic published when a goal is sent,
captures the points, and prints a per-segment velocity analysis.
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
import math, time, sys


class TrajectoryDiag(Node):
    def __init__(self):
        super().__init__("trajectory_diag")

        # Subscribe to the trajectory the controller receives
        self.traj_sub = self.create_subscription(
            JointTrajectory,
            "/nakul_scaled_joint_trajectory_controller/joint_trajectory",
            self.on_trajectory,
            10,
        )
        # Subscribe to controller state to watch actual vs desired during execution
        self.state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            "/nakul_scaled_joint_trajectory_controller/controller_state",
            self.on_state,
            10,
        )
        self.got_traj = False
        self.state_samples = []
        self.capturing_state = False
        self.capture_start = None
        self.get_logger().info("Waiting for a trajectory on /nakul_scaled_joint_trajectory_controller/joint_trajectory ...")

    def on_trajectory(self, msg: JointTrajectory):
        if self.got_traj:
            return
        self.got_traj = True
        pts = msg.points
        n = len(pts)
        self.get_logger().info(f"\n{'='*80}")
        self.get_logger().info(f"TRAJECTORY RECEIVED: {n} points, joints={msg.joint_names}")

        # Analyze every segment
        max_vel_overall = 0.0
        max_accel_overall = 0.0
        vel_histogram = {}  # bucket velocities
        jerk_events = []

        self.get_logger().info(f"\n{'Seg':>4} | {'dt(ms)':>8} | {'max_dq/dt':>10} | {'joint':>6} | {'vel_field_max':>13} | {'accel_field_max':>15} | {'has_vel':>7} | {'has_acc':>7}")
        self.get_logger().info("-" * 100)

        prev_vels = [0.0] * 6  # track velocity changes for jerk detection

        for k in range(1, n):
            t_prev = pts[k-1].time_from_start.sec + pts[k-1].time_from_start.nanosec * 1e-9
            t_curr = pts[k].time_from_start.sec + pts[k].time_from_start.nanosec * 1e-9
            dt = t_curr - t_prev

            if dt <= 0:
                self.get_logger().warn(f"Seg {k}: dt={dt:.6f} <= 0!")
                continue

            # Compute actual velocity from position differences
            max_dq_dt = 0.0
            max_joint = -1
            nj = min(len(pts[k].positions), len(pts[k-1].positions))
            for j in range(nj):
                dq = abs(pts[k].positions[j] - pts[k-1].positions[j])
                v = dq / dt
                if v > max_dq_dt:
                    max_dq_dt = v
                    max_joint = j

            # Check velocity fields
            vel_max = 0.0
            has_vel = len(pts[k].velocities) > 0
            if has_vel:
                vel_max = max(abs(v) for v in pts[k].velocities)

            # Check acceleration fields
            acc_max = 0.0
            has_acc = len(pts[k].accelerations) > 0
            if has_acc:
                acc_max = max(abs(a) for a in pts[k].accelerations)

            max_vel_overall = max(max_vel_overall, max_dq_dt)
            max_accel_overall = max(max_accel_overall, acc_max)

            # Velocity bucket (round to 0.01)
            bucket = round(max_dq_dt, 2)
            vel_histogram[bucket] = vel_histogram.get(bucket, 0) + 1

            # Detect jerk: sudden velocity change
            if has_vel and k > 1:
                for j in range(nj):
                    if j < len(pts[k].velocities) and j < len(pts[k-1].velocities):
                        dv = abs(pts[k].velocities[j] - pts[k-1].velocities[j])
                        if dv > 0.05:  # significant velocity jump
                            jerk_events.append((k, j, dv, dt))

            # Print first 20, last 5, and any anomalous segments
            is_anomalous = max_dq_dt > 0.2 or dt < 0.01 or dt > 0.05 or (has_vel and vel_max > 0.2)
            if k <= 20 or k >= n - 5 or is_anomalous or k % 50 == 0:
                self.get_logger().info(
                    f"{k:4d} | {dt*1000:8.2f} | {max_dq_dt:10.4f} | J{max_joint:5d} | {vel_max:13.4f} | {acc_max:15.4f} | {str(has_vel):>7} | {str(has_acc):>7}"
                    + (" *** ANOMALOUS" if is_anomalous else "")
                )

        self.get_logger().info(f"\n{'='*80}")
        self.get_logger().info(f"SUMMARY:")
        self.get_logger().info(f"  Total points: {n}")
        total_dur = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9
        self.get_logger().info(f"  Total duration: {total_dur:.3f}s")
        self.get_logger().info(f"  Max position-derived velocity: {max_vel_overall:.4f} rad/s")
        self.get_logger().info(f"  Max acceleration field: {max_accel_overall:.4f} rad/s²")
        self.get_logger().info(f"  Jerk events (dv > 0.05 rad/s between consecutive points): {len(jerk_events)}")
        if jerk_events:
            self.get_logger().info(f"  Top 10 jerk events:")
            for seg, joint, dv, dt in sorted(jerk_events, key=lambda x: -x[2])[:10]:
                self.get_logger().info(f"    Seg {seg}, Joint {joint}: dv={dv:.4f} rad/s in dt={dt*1000:.1f}ms")

        # Velocity distribution
        self.get_logger().info(f"\n  Velocity distribution (position-derived):")
        for v in sorted(vel_histogram.keys()):
            count = vel_histogram[v]
            bar = "#" * min(count, 50)
            self.get_logger().info(f"    {v:.2f} rad/s: {count:4d} {bar}")

        # Start capturing controller state for execution analysis
        self.capturing_state = True
        self.capture_start = time.time()
        self.get_logger().info(f"\nNow capturing controller state for 30s to analyze execution tracking...")

    def on_state(self, msg: JointTrajectoryControllerState):
        if not self.capturing_state:
            return
        elapsed = time.time() - self.capture_start
        if elapsed > 30:
            self.capturing_state = False
            self._analyze_state_samples()
            return

        # Sample every ~100ms
        if len(self.state_samples) > 0:
            last_t = self.state_samples[-1][0]
            if elapsed - last_t < 0.1:
                return

        # Record (time, desired_pos, actual_pos, desired_vel, actual_vel)
        self.state_samples.append((
            elapsed,
            list(msg.reference.positions),
            list(msg.feedback.positions),
            list(msg.reference.velocities) if msg.reference.velocities else [0]*6,
            list(msg.feedback.velocities) if msg.feedback.velocities else [0]*6,
        ))

    def _analyze_state_samples(self):
        if len(self.state_samples) < 2:
            self.get_logger().info("Not enough state samples captured.")
            return

        self.get_logger().info(f"\n{'='*80}")
        self.get_logger().info(f"CONTROLLER STATE ANALYSIS ({len(self.state_samples)} samples)")

        max_pos_error = [0.0] * 6
        max_vel_error = [0.0] * 6
        max_actual_vel = [0.0] * 6

        for i in range(1, len(self.state_samples)):
            t, des_p, act_p, des_v, act_v = self.state_samples[i]
            for j in range(6):
                pe = abs(des_p[j] - act_p[j])
                if pe > max_pos_error[j]:
                    max_pos_error[j] = pe

                ve = abs(des_v[j] - act_v[j])
                if ve > max_vel_error[j]:
                    max_vel_error[j] = ve

                if abs(act_v[j]) > max_actual_vel[j]:
                    max_actual_vel[j] = abs(act_v[j])

        self.get_logger().info(f"  Max position tracking error per joint (rad):")
        for j in range(6):
            self.get_logger().info(f"    J{j}: {max_pos_error[j]:.6f} rad ({math.degrees(max_pos_error[j]):.4f}°)")

        self.get_logger().info(f"  Max velocity tracking error per joint (rad/s):")
        for j in range(6):
            self.get_logger().info(f"    J{j}: {max_vel_error[j]:.4f} rad/s")

        self.get_logger().info(f"  Max actual velocity per joint (rad/s):")
        for j in range(6):
            self.get_logger().info(f"    J{j}: {max_actual_vel[j]:.4f} rad/s")

        # Detect jerks in actual position (sudden changes in velocity)
        jerk_count = 0
        for i in range(2, len(self.state_samples)):
            t0 = self.state_samples[i-2][0]
            t1 = self.state_samples[i-1][0]
            t2 = self.state_samples[i][0]
            dt1 = t1 - t0
            dt2 = t2 - t1
            if dt1 <= 0 or dt2 <= 0:
                continue
            for j in range(6):
                v1 = (self.state_samples[i-1][2][j] - self.state_samples[i-2][2][j]) / dt1
                v2 = (self.state_samples[i][2][j] - self.state_samples[i-1][2][j]) / dt2
                accel = (v2 - v1) / ((dt1 + dt2) / 2)
                if abs(accel) > 2.0:  # significant acceleration spike
                    jerk_count += 1
                    if jerk_count <= 10:
                        self.get_logger().info(f"  JERK: t={t2:.2f}s J{j} accel={accel:.2f} rad/s² (v1={v1:.4f} → v2={v2:.4f})")

        self.get_logger().info(f"  Total jerk events (|accel| > 2.0): {jerk_count}")
        self.get_logger().info(f"\nDiagnostic complete.")
        sys.exit(0)


def main():
    rclpy.init()
    node = TrajectoryDiag()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
