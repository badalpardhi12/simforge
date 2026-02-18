"""
Protocol Executor — runs a multi-pose protocol (proto-sim).

Orchestrates:
  1. (optional) Home the robot
  2. Plan a multi-waypoint trajectory via cuRobo
  3. Execute via the trajectory executor
  4. Report per-pose progress over the WebSocket
  5. (optional) Home the robot afterwards
"""

import asyncio
import json
import math
import time
import traceback
from typing import List, Optional, Tuple

import websockets

from .config import ROBOT_CONFIG, RTDE_AVAILABLE, RobotStateInfo


# ── Pose-comparison helpers ──────────────────────────────────────


def _axis_angle_to_quat(rx: float, ry: float, rz: float) -> List[float]:
    """Convert axis-angle rotation vector to [qw, qx, qy, qz]."""
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]
    half = angle / 2.0
    s = math.sin(half) / angle
    return [math.cos(half), rx * s, ry * s, rz * s]


def _quat_angular_distance(q1: List[float], q2: List[float]) -> float:
    """Return the angular difference (degrees) between two quaternions.

    Both quaternions are [qw, qx, qy, qz].
    """
    dot = sum(a * b for a, b in zip(q1, q2))
    dot = max(-1.0, min(1.0, abs(dot)))  # abs handles double-cover
    return 2.0 * math.acos(dot) * 180.0 / math.pi


def _position_error_mm(
    actual: List[float], target: List[float],
) -> Tuple[float, List[float]]:
    """Return (RSS error in mm, [dx, dy, dz] in mm)."""
    errs = [(a - t) * 1000.0 for a, t in zip(actual[:3], target[:3])]
    rss = math.sqrt(sum(e * e for e in errs))
    return rss, errs


class ProtocolExecutor:
    """Runs a multi-pose protocol on a single robot."""

    def __init__(self, node, planner, executor, joint_state_mgr):
        """
        Parameters
        ----------
        node : CommandGatewayNode
        planner : CuroboPlanner
        executor : TrajectoryExecutor
        joint_state_mgr : JointStateManager
        """
        self._node = node
        self._log = node.get_logger()
        self._planner = planner
        self._executor = executor
        self._js_mgr = joint_state_mgr

        # Flags — set by node-level stop / estop handlers
        self.running = False
        self.stop_requested = False

    # ── Entry point (called from RPC handler) ────────────────────

    async def run(
        self, client, request_id: str,
        robot_name: str, poses: list,
        idle_time: float = 2.0,
        mode: str = "simulation",
        move_speed: float = 0.5,
        go_home_before: bool = False,
        go_home_after: bool = False,
    ):
        """Top-level coroutine — wraps ``_run_inner`` with error handling."""
        self.running = True
        self.stop_requested = False

        cfg = ROBOT_CONFIG[robot_name]
        total = len(poses)

        try:
            await self._run_inner(
                client, request_id, robot_name, poses,
                idle_time, mode, move_speed, cfg, total,
                go_home_before, go_home_after,
            )
        except websockets.ConnectionClosed:
            self._log.warn(
                "Client disconnected during proto_sim — aborting"
            )
            self.stop_requested = True
        except Exception as e:
            self._log.error(f"Proto-sim error: {e}")
            traceback.print_exc()
            try:
                await self._send_error(
                    client, request_id,
                    f"Protocol execution failed: {e}",
                )
            except Exception:
                pass  # client may have disconnected
        finally:
            self.running = False

    # ── Inner implementation ─────────────────────────────────────


    async def _run_inner(
        self, client, request_id, robot_name, poses,
        idle_time, mode, move_speed, cfg, total,
        go_home_before, go_home_after,
    ):
        self._log.info(
            f"Proto-sim START (cuRobo sequential): {total} poses on "
            f"{robot_name} (mode={mode}, speed={move_speed}, "
            f"idle={idle_time}s, home_before={go_home_before}, "
            f"home_after={go_home_after})"
        )

        # ── Optional home-before ─────────────────────────────────
        if go_home_before:
            home_ok, home_err = await self._home_if_needed(
                robot_name, cfg, mode,
            )
            if not home_ok:
                self._log.error(
                    f"Home-before failed for {robot_name}: {home_err}"
                )
                await self._send_error(
                    client, request_id,
                    f"Cannot move {robot_name} to home position: "
                    f"{home_err}. Clear the fault on the teach pendant "
                    f"and try again.",
                )
                return

        # ── Pre-flight: check robot is controllable ──────────────
        if mode in ("real", "both") and RTDE_AVAILABLE:
            rtde = self._executor._rtde.get(robot_name)
            if rtde is not None:
                # Clear stale errors from any previous protocol run
                # so we check the *current* robot state, not a cached
                # failure message from a prior session.
                rtde.last_error = ""

                # Reconnect the receive interface if it's dead — it
                # may have broken during a previous fault or servoJ.
                if not rtde.is_connected:
                    self._log.info(
                        f"RTDE recv down for {robot_name} "
                        f"— reconnecting for pre-flight…"
                    )
                    rtde.reconnect_receive()

                # Live robot mode check (mode 7 = RUNNING)
                try:
                    robot_mode = (
                        rtde._recv.getRobotMode()
                        if rtde._recv else -1
                    )
                except Exception:
                    robot_mode = -1

                if robot_mode != 7 and robot_mode != -1:
                    err = (
                        f"Robot mode is {robot_mode} "
                        f"(need 7/RUNNING). Start the robot "
                        f"program on the teach pendant first."
                    )
                    self._log.error(
                        f"RTDE pre-flight failed for "
                        f"{robot_name}: {err}"
                    )
                    await self._send_error(
                        client, request_id,
                        f"Robot {robot_name} is not "
                        f"controllable: {err}",
                    )
                    return

        # ── Ensure robot ready ───────────────────────────────────
        ready = await self._executor.ensure_robot_ready(
            robot_name, timeout=15.0
        )
        if not ready:
            cur_mode = getattr(self._node, '_current_mode', 'simulation')
            if cur_mode in ("real", "both"):
                await self._send_error(
                    client, request_id,
                    f"Robot program not running on {robot_name}"
                )
                return

        # ── Sequential plan-execute per segment ──────────────────
        # IMPORTANT: Planning and execution must NEVER overlap!
        # Both run in separate OS threads (asyncio.to_thread /
        # run_in_executor).  If they run concurrently, Python GIL
        # contention starves the servoJ 500 Hz loop down to ~12 Hz,
        # causing extremely jerky robot motion.
        #
        # Sequence per pose:
        #   1. Plan  (CUDA in background thread, event loop free)
        #   2. Execute (servoJ in executor thread, event loop free)
        #   3. Dwell  (asyncio.sleep, event loop free)

        hw_abort = False
        completed = 0
        ik_failed = 0
        exec_failed = 0
        position_errors = []    # RSS joint-space error (deg) per pose
        cartesian_errors = []   # (pos_mm, orient_deg) per pose

        current_joints = (
            self._js_mgr.robot_states[robot_name].joint_positions
        )
        if not current_joints or len(current_joints) != 6:
            current_joints = list(cfg["home_position"])

        for i, pose_data in enumerate(poses):
            if self.stop_requested:
                break

            pose_name = pose_data.get("name", f"pose_{i}")
            position = pose_data.get("position", [0, 0, 0])
            orientation = pose_data.get("orientation", [0, 0, 0, 1])

            # ── Hardware fault check ─────────────────────────────
            hw_issues = self._detect_hardware_issues()
            if hw_issues:
                # Safeguard stops are recoverable — pause & wait
                if not self._has_fatal_hardware_issue(hw_issues):
                    self._log.warn(
                        f"Safeguard stop before pose {i+1}: "
                        f"{', '.join(hw_issues)} — pausing…"
                    )
                    await self._send_feedback(
                        client, request_id, i, total,
                        (i / max(total, 1)) * 100, "paused",
                        message=(
                            f"⏸ SAFEGUARD STOP: "
                            f"{', '.join(hw_issues)}. "
                            f"Waiting for area to clear…"
                        ),
                    )
                    # Wait for all safeguard-stopped robots to clear
                    all_clear = await self._wait_safeguard_clear_all()
                    if not all_clear:
                        hw_abort = True
                        self._log.error(
                            "Safeguard did not clear — aborting"
                        )
                        await self._send_feedback(
                            client, request_id, i, total,
                            (i / max(total, 1)) * 100, "error",
                            message=(
                                "⚠ Safeguard stop did not clear "
                                "within timeout — aborting protocol."
                            ),
                        )
                        self.stop_requested = True
                        break
                    self._log.info(
                        "Safeguard cleared — resuming protocol"
                    )
                    await self._send_feedback(
                        client, request_id, i, total,
                        (i / max(total, 1)) * 100, "resuming",
                        message="✓ Safeguard cleared — resuming…",
                    )
                else:
                    # Fatal (protective/emergency stop, mode not
                    # RUNNING, etc.) — pause and wait for the user
                    # to reset the robot & put it back in remote
                    # control, rather than aborting immediately.
                    self._log.error(
                        f"Hardware fault at pose {i+1}: "
                        f"{', '.join(hw_issues)} — pausing for "
                        f"recovery (press Stop to abort)"
                    )
                    await self._send_feedback(
                        client, request_id, i, total,
                        (i / max(total, 1)) * 100, "fault_paused",
                        message=(
                            f"⚠ HARDWARE FAULT at pose {i+1}/"
                            f"{total}: {', '.join(hw_issues)}. "
                            f"Reset the robot and put it back in "
                            f"remote control — the protocol will "
                            f"resume automatically. "
                            f"Press Stop to abort."
                        ),
                    )

                    recovered = await self._wait_for_robot_recovery(
                        robot_name, client, request_id, i, total,
                    )

                    if recovered and not self.stop_requested:
                        self._log.info(
                            f"Robot {robot_name} recovered — "
                            f"resuming protocol from pose "
                            f"{i+1}/{total}"
                        )
                        await self._send_feedback(
                            client, request_id, i, total,
                            (i / max(total, 1)) * 100, "resuming",
                            message=(
                                f"✓ Robot recovered — resuming "
                                f"from pose {i+1}/{total}…"
                            ),
                        )
                        # Fall through to the planning step for
                        # the current pose (do NOT break or skip)
                    else:
                        hw_abort = True
                        reason = (
                            "user stopped"
                            if self.stop_requested
                            else "recovery timed out"
                        )
                        self._log.error(
                            f"Recovery failed ({reason}) — "
                            f"aborting protocol at pose "
                            f"{i+1}/{total}"
                        )
                        await self._send_feedback(
                            client, request_id, i, total,
                            (i / max(total, 1)) * 100, "error",
                            message=(
                                f"⚠ Protocol aborted at pose "
                                f"{i+1}/{total}: {reason}."
                            ),
                        )
                        self.stop_requested = True
                        break

            # ── Plan ─────────────────────────────────────────────
            pct = (i / total) * 100
            await self._send_feedback(
                client, request_id, i, total, pct,
                "planning",
                f"Planning {i+1}/{total}: {pose_name}",
            )

            t0 = time.monotonic()
            try:
                result = await self._planner.plan_to_pose(
                    robot_name, position, orientation,
                    current_joints=current_joints,
                    velocity_scaling=move_speed,
                )
            except Exception as e:
                self._log.error(
                    f"Planning exception for {pose_name}: {e}"
                )
                result = None
            plan_dt = time.monotonic() - t0

            if result is None:
                ik_failed += 1
                self._log.warn(
                    f"  [{i+1}/{total}] \u2717 plan failed "
                    f"{pose_name} ({plan_dt:.1f}s)"
                )
                continue

            ros_traj = self._planner.result_to_ros_trajectory(
                result, robot_name,
            )
            if ros_traj is None:
                ik_failed += 1
                continue

            # Planned final position (fallback if actual read fails)
            last_pt = ros_traj.joint_trajectory.points[-1]
            planned_final = list(last_pt.positions)

            self._log.info(
                f"  [{i+1}/{total}] planned "
                f"{pose_name} ({plan_dt:.1f}s)"
            )

            # ── Execute ──────────────────────────────────────────
            await self._send_feedback(
                client, request_id, i, total, pct,
                "moving", current_pose_name=pose_name,
            )

            pts = ros_traj.joint_trajectory.points
            traj_dur = (
                pts[-1].time_from_start.sec
                + pts[-1].time_from_start.nanosec * 1e-9
            ) if pts else 5.0
            exec_timeout = max(traj_dur * 2.0, 30.0)

            ok = await self._executor.execute(
                robot_name, ros_traj, timeout=exec_timeout,
            )

            # ── Safeguard-aware retry ────────────────────────────
            # If execution failed and the robot is in (or was just
            # in) a safeguard stop, wait for clearance and retry
            # the same trajectory once rather than counting it as
            # a permanent failure.
            if not ok and RTDE_AVAILABLE:
                rtde = self._executor._rtde.get(robot_name)
                if rtde is not None and rtde.is_safeguard_stopped():
                    self._log.warn(
                        f"  [{i+1}/{total}] exec interrupted by "
                        f"safeguard stop on {pose_name} — "
                        f"waiting for clearance…"
                    )
                    await self._send_feedback(
                        client, request_id, i, total, pct,
                        "paused",
                        message=(
                            f"⏸ Safeguard stop during {pose_name} "
                            f"— waiting for area to clear…"
                        ),
                    )
                    all_clear = await self._wait_safeguard_clear_all()
                    if all_clear:
                        self._log.info(
                            f"  [{i+1}/{total}] safeguard cleared "
                            f"— retrying {pose_name}"
                        )
                        await self._send_feedback(
                            client, request_id, i, total, pct,
                            "moving",
                            message=(
                                f"▶ Retrying {pose_name} after "
                                f"safeguard clear"
                            ),
                            current_pose_name=pose_name,
                        )
                        ok = await self._executor.execute(
                            robot_name, ros_traj, timeout=exec_timeout,
                        )

            if ok:
                completed += 1

                # ── Position verification & current_joints update ─
                # Use the robot's ACTUAL post-execution position as
                # the start for the next trajectory, eliminating the
                # jump caused by planning from the (imprecise) last
                # planned waypoint.
                actual_q = self._executor.get_actual_position(
                    robot_name,
                )
                if actual_q is not None:
                    current_joints = list(actual_q)
                    # Log joint-space error (planned vs actual)
                    errors_deg = [
                        (actual_q[j] - planned_final[j])
                        * 180.0 / math.pi
                        for j in range(len(planned_final))
                    ]
                    rss_deg = math.sqrt(
                        sum(e ** 2 for e in errors_deg)
                    )
                    position_errors.append(rss_deg)
                    self._log.info(
                        f"  [{i+1}/{total}] joint error "
                        f"{pose_name}: "
                        f"RSS={rss_deg*1000:.0f}mDeg  "
                        f"per-joint(mDeg)="
                        f"{[round(e*1000,1) for e in errors_deg]}"
                    )
                else:
                    # Fallback: use planned final (old behaviour)
                    current_joints = list(planned_final)
                    self._log.warn(
                        f"  [{i+1}/{total}] could not read actual "
                        f"position — using planned final as start "
                        f"for next segment"
                    )

                # ── Cartesian pose verification ──────────────────
                # Compare the target Cartesian pose (what we asked
                # the planner to reach) with the actual TCP pose
                # from two sources:
                #   1. RTDE getActualTCPPose() — ground truth from
                #      the UR controller
                #   2. cuRobo FK on the actual joints — verifies
                #      model consistency
                #
                # FRAME NOTE:
                #   • Client sends target poses in the robot's
                #     base_link frame.
                #   • cuRobo's MotionGen interprets goal poses in
                #     its own "world" frame, which — for a multi-
                #     robot cell — includes the world→base_link
                #     transform from the URDF.  So the target
                #     pose (base_link) and cuRobo FK output
                #     (cuRobo world) are in the SAME frame only
                #     if that transform is identity.
                #   • RTDE getActualTCPPose() returns TCP in the
                #     robot's base_link frame — different from
                #     cuRobo's world frame when the base is offset
                #     (e.g. nakul_ur5e at [-0.6758, 0, 1.03]).
                #
                # Therefore:
                #   • FK vs target: both in cuRobo world frame →
                #     direct comparison is correct.
                #   • RTDE TCP vs target: different frames when
                #     base is offset → DO NOT compare directly.
                #   • FK vs RTDE TCP: different frames → removed.

                # FK from actual joints (cuRobo world frame)
                fk_result = None
                if actual_q is not None and self._planner is not None:
                    fk_result = self._planner.compute_fk(
                        robot_name, actual_q,
                    )

                if fk_result is not None:
                    fk_pos, fk_quat = fk_result

                    # Target pose (base_link / cuRobo world frame)
                    target_pos = position     # [x, y, z] metres
                    target_quat_xyzw = orientation  # [qx,qy,qz,qw]
                    target_quat = [
                        target_quat_xyzw[3],
                        target_quat_xyzw[0],
                        target_quat_xyzw[1],
                        target_quat_xyzw[2],
                    ]  # → [qw, qx, qy, qz]

                    pos_err_mm, pos_xyz_mm = _position_error_mm(
                        fk_pos, target_pos,
                    )
                    orient_err_deg = _quat_angular_distance(
                        fk_quat, target_quat,
                    )
                    cartesian_errors.append((pos_err_mm, orient_err_deg))

                    self._log.info(
                        f"  [{i+1}/{total}] CARTESIAN ERROR "
                        f"({pose_name}): "
                        f"pos={pos_err_mm:.2f}mm "
                        f"[dx={pos_xyz_mm[0]:.2f}, "
                        f"dy={pos_xyz_mm[1]:.2f}, "
                        f"dz={pos_xyz_mm[2]:.2f}]mm  "
                        f"orient={orient_err_deg:.3f}°"
                    )
                    if pos_err_mm > 5.0:
                        self._log.warn(
                            f"  ⚠ LARGE Cartesian error on "
                            f"{robot_name}: {pos_err_mm:.1f}mm"
                        )

                self._log.info(
                    f"  [{i+1}/{total}] \u2713 {pose_name}"
                )
                await self._send_feedback(
                    client, request_id, i, total,
                    ((i + 1) / total) * 100,
                    "reached", current_pose_name=pose_name,
                )
                # Dwell
                if idle_time > 0 and i < len(poses) - 1:
                    await asyncio.sleep(idle_time)
            else:
                # ── Fault-aware recovery after exec failure ──────
                # If execution failed due to a hardware fault (robot
                # mode not RUNNING, protective stop, etc.), pause and
                # wait for the user to reset the robot rather than
                # just skipping this pose and letting the next
                # iteration's hardware check abort the protocol.
                if RTDE_AVAILABLE and not self.stop_requested:
                    post_hw = self._detect_hardware_issues()
                    if post_hw and self._has_fatal_hardware_issue(
                        post_hw,
                    ):
                        self._log.error(
                            f"  [{i+1}/{total}] exec failed "
                            f"{pose_name} due to hardware fault: "
                            f"{', '.join(post_hw)} — pausing for "
                            f"recovery"
                        )
                        await self._send_feedback(
                            client, request_id, i, total,
                            (i / max(total, 1)) * 100,
                            "fault_paused",
                            message=(
                                f"⚠ HARDWARE FAULT during pose "
                                f"{i+1}/{total} ({pose_name}): "
                                f"{', '.join(post_hw)}. "
                                f"Reset the robot and put it back "
                                f"in remote control — will retry "
                                f"this pose automatically. "
                                f"Press Stop to abort."
                            ),
                        )
                        recovered = (
                            await self._wait_for_robot_recovery(
                                robot_name, client, request_id,
                                i, total,
                            )
                        )
                        if recovered and not self.stop_requested:
                            self._log.info(
                                f"Robot recovered — retrying pose "
                                f"{i+1} ({pose_name})"
                            )
                            await self._send_feedback(
                                client, request_id, i, total,
                                (i / max(total, 1)) * 100,
                                "resuming",
                                message=(
                                    f"✓ Robot recovered — retrying "
                                    f"pose {i+1}/{total}…"
                                ),
                            )
                            # Retry execution of the same trajectory
                            ok = await self._executor.execute(
                                robot_name, ros_traj,
                                timeout=exec_timeout,
                            )
                            if ok:
                                completed += 1
                                actual_q = (
                                    self._executor.get_actual_position(
                                        robot_name,
                                    )
                                )
                                if actual_q is not None:
                                    current_joints = list(actual_q)
                                else:
                                    current_joints = list(
                                        planned_final
                                    )
                                self._log.info(
                                    f"  [{i+1}/{total}] ✓ "
                                    f"{pose_name} (after recovery)"
                                )
                                await self._send_feedback(
                                    client, request_id, i, total,
                                    ((i + 1) / total) * 100,
                                    "reached",
                                    current_pose_name=pose_name,
                                )
                                if (idle_time > 0
                                        and i < len(poses) - 1):
                                    await asyncio.sleep(idle_time)
                                continue  # next pose
                            # Retry also failed — fall through
                        else:
                            hw_abort = True
                            self.stop_requested = True
                            break

                exec_failed += 1
                self._log.error(
                    f"  [{i+1}/{total}] \u2717 exec failed "
                    f"{pose_name}"
                )
                # Even on failure, update current_joints from actual
                # position so the next pose starts from the right
                # place (the robot may have partially executed).
                actual_q = self._executor.get_actual_position(
                    robot_name,
                )
                if actual_q is not None:
                    current_joints = list(actual_q)

        # ── Optional home-after ──────────────────────────────────
        if go_home_after and completed > 0 and not self.stop_requested:
            self._log.info(f"Returning {robot_name} to home\u2026")
            home_ok, home_err = await self._home_if_needed(
                robot_name, cfg, mode,
            )
            if not home_ok:
                self._log.warn(
                    f"Home-after failed for {robot_name}: {home_err}"
                )

        self.running = False

        # ── Build result ─────────────────────────────────────────
        summary_parts = [
            f"Completed {completed}/{total} poses (cuRobo sequential)"
        ]
        if ik_failed:
            summary_parts.append(f"{ik_failed} IK/plan failed")
        if exec_failed:
            summary_parts.append(f"{exec_failed} execution failed")

        # Position accuracy stats
        if position_errors:
            avg_err = sum(position_errors) / len(position_errors)
            max_err = max(position_errors)
            summary_parts.append(
                f"accuracy: avg={avg_err*1000:.0f}mDeg "
                f"max={max_err*1000:.0f}mDeg"
            )
            self._log.info(
                f"Position accuracy summary for {robot_name}: "
                f"{len(position_errors)} measurements, "
                f"avg RSS error={avg_err*1000:.1f}mDeg, "
                f"max RSS error={max_err*1000:.1f}mDeg, "
                f"all errors(mDeg)="
                f"{[round(e*1000,1) for e in position_errors]}"
            )

        # Cartesian accuracy stats
        if cartesian_errors:
            pos_errs = [e[0] for e in cartesian_errors]
            orient_errs = [e[1] for e in cartesian_errors]
            avg_pos = sum(pos_errs) / len(pos_errs)
            max_pos = max(pos_errs)
            avg_orient = sum(orient_errs) / len(orient_errs)
            max_orient = max(orient_errs)
            summary_parts.append(
                f"pose: avg={avg_pos:.1f}mm/{avg_orient:.2f}° "
                f"max={max_pos:.1f}mm/{max_orient:.2f}°"
            )
            self._log.info(
                f"Cartesian accuracy summary for {robot_name}: "
                f"{len(cartesian_errors)} measurements, "
                f"position: avg={avg_pos:.2f}mm max={max_pos:.2f}mm, "
                f"orientation: avg={avg_orient:.3f}° "
                f"max={max_orient:.3f}°"
            )
            self._log.info(
                f"  per-pose pos(mm): "
                f"{[round(e[0], 2) for e in cartesian_errors]}"
            )
            self._log.info(
                f"  per-pose orient(deg): "
                f"{[round(e[1], 3) for e in cartesian_errors]}"
            )

        hardware_issues = self._detect_hardware_issues()
        if hardware_issues:
            summary_parts.append(
                f"\u26a0 Hardware: {', '.join(hardware_issues)}"
            )

        summary = " | ".join(summary_parts)
        self._log.info(f"Proto-sim DONE: {summary}")

        await client.websocket.send(json.dumps({
            "type": "rpc_result",
            "request_id": request_id,
            "success": completed > 0 and not hw_abort,
            "message": summary,
            "completed": completed,
            "collision_rejected": 0,
            "ik_failed": ik_failed,
            "real_failed": exec_failed,
            "total": total,
            "stopped": self.stop_requested,
            "hardware_issues": hardware_issues,
            "hardware_abort": hw_abort,
            "accuracy": {
                "joint_errors_mdeg": (
                    [round(e * 1000, 1) for e in position_errors]
                    if position_errors else []
                ),
                "cartesian_pos_mm": (
                    [round(e[0], 2) for e in cartesian_errors]
                    if cartesian_errors else []
                ),
                "cartesian_orient_deg": (
                    [round(e[1], 3) for e in cartesian_errors]
                    if cartesian_errors else []
                ),
            },
        }))


    # ── Pose-by-pose fallback ─────────────────────────────────────

    async def _run_pose_by_pose(
        self, client, request_id, robot_name, poses,
        idle_time, mode, move_speed, cfg, total,
    ):
        """Pose-by-pose cuRobo fallback."""
        completed, ik_failed, plan_failed = 0, 0, 0
        self._log.info("Running pose-by-pose execution (fallback)")

        for i, pose_data in enumerate(poses):
            if self.stop_requested:
                break
            pose_name = pose_data.get("name", f"pose_{i}")
            position = pose_data.get("position", [0, 0, 0])
            orientation = pose_data.get("orientation", [0, 0, 0, 1])

            await self._send_feedback(
                client, request_id, i, total,
                (i / total) * 100, "moving",
            )

            current = self._js_mgr.robot_states[robot_name].joint_positions
            if not current or len(current) != 6:
                current = list(cfg["home_position"])

            result = await self._planner.plan_to_pose(
                robot_name, position, orientation,
                current_joints=current,
                velocity_scaling=move_speed,
            )
            if result is None:
                plan_failed += 1
                self._log.warn(
                    f"  [{i+1}/{total}] ✗ plan failed {pose_name}"
                )
                continue

            ros_traj = self._planner.result_to_ros_trajectory(
                result, robot_name
            )
            if ros_traj is None:
                plan_failed += 1
                continue

            ok = await self._executor.execute(robot_name, ros_traj)
            if ok:
                completed += 1
                self._log.info(f"  [{i+1}/{total}] ✓ {pose_name}")
                await asyncio.sleep(idle_time)
            else:
                plan_failed += 1
                self._log.error(
                    f"  [{i+1}/{total}] ✗ exec failed {pose_name}"
                )

        self.running = False
        summary = f"Completed {completed}/{total} poses (pose-by-pose fallback)"
        hardware_issues = self._detect_hardware_issues()

        await client.websocket.send(json.dumps({
            "type": "rpc_result",
            "request_id": request_id,
            "success": completed > 0,
            "message": summary,
            "completed": completed,
            "collision_rejected": 0,
            "ik_failed": ik_failed,
            "real_failed": plan_failed,
            "total": total,
            "stopped": self.stop_requested,
            "hardware_issues": hardware_issues,
        }))

    # ── Helpers ──────────────────────────────────────────────────

    async def _wait_safeguard_clear_all(
        self, timeout: float = float('inf'),
    ) -> bool:
        """Wait for all RTDE robots to exit safeguard stop.

        The default timeout is infinite — the system waits as long as
        the safety zone is occupied.  Runs the blocking
        ``wait_for_safeguard_clear()`` in a thread so the asyncio
        event loop stays responsive (e.g. for WebSocket keep-alive).
        """
        loop = asyncio.get_event_loop()
        for rn, rtde in self._executor._rtde.items():
            if rtde.is_safeguard_stopped():
                cleared = await loop.run_in_executor(
                    None,
                    lambda r=rtde: r.wait_for_safeguard_clear(
                        timeout=timeout, logger=self._log,
                    ),
                )
                if not cleared:
                    return False
        return True

    async def _wait_for_robot_recovery(
        self,
        robot_name: str,
        client,
        request_id: str,
        pose_idx: int,
        total: int,
        poll_interval: float = 3.0,
        feedback_interval: float = 15.0,
    ) -> bool:
        """Wait for the robot to recover from a hardware fault.

        Polls the RTDE receive interface to detect when the robot
        returns to mode 7 (RUNNING).  The user must:
          1. Clear the fault on the teach pendant
          2. Restart the robot program (put in remote control)

        While waiting, periodic ``fault_paused`` feedback is sent to
        the client so the WebSocket stays alive and the user sees
        that the server is still listening.

        Returns
        -------
        True  — robot recovered (mode 7, RTDE reconnected)
        False — ``stop_requested`` was set or recovery failed
        """
        rtde = self._executor._rtde.get(robot_name)
        if rtde is None:
            return False

        last_feedback = time.monotonic()
        attempt = 0

        while not self.stop_requested:
            attempt += 1
            await asyncio.sleep(poll_interval)

            if self.stop_requested:
                return False

            # Try to reconnect the receive interface if it's dead
            if not rtde.is_connected:
                self._log.info(
                    f"Recovery [{robot_name}]: RTDE recv down — "
                    f"attempting reconnect (attempt {attempt})…"
                )
                ok = await asyncio.get_event_loop().run_in_executor(
                    None, rtde.reconnect_receive,
                )
                if not ok:
                    # Send periodic feedback so client stays alive
                    now = time.monotonic()
                    if now - last_feedback >= feedback_interval:
                        last_feedback = now
                        await self._send_feedback(
                            client, request_id, pose_idx, total,
                            (pose_idx / max(total, 1)) * 100,
                            "fault_paused",
                            message=(
                                f"⏳ Waiting for {robot_name} to "
                                f"come back online… (attempt "
                                f"{attempt}, RTDE reconnect failed)"
                            ),
                        )
                    continue

            # Recv is connected — check robot mode
            try:
                robot_mode = rtde._recv.getRobotMode()
            except Exception:
                robot_mode = -1

            if robot_mode == 7:
                # Robot mode is RUNNING — now also verify the
                # safety controller has cleared.  The UR can
                # report mode 7 while the safety system is still
                # in PROTECTIVE_STOP (different registers).
                sm = rtde.get_safety_mode()
                if sm in (
                    rtde.SAFETY_MODE_NORMAL,
                    rtde.SAFETY_MODE_REDUCED,
                ):
                    # Fully recovered — clear stale errors
                    rtde.last_error = ""
                    self._log.info(
                        f"Recovery [{robot_name}]: robot mode=7, "
                        f"safety_mode={sm} (NORMAL/REDUCED) — "
                        f"recovery successful (attempt {attempt})"
                    )
                    return True
                else:
                    # Mode is 7 but safety hasn't cleared yet.
                    # Common: UR briefly shows mode 7 while
                    # still in protective stop.
                    now = time.monotonic()
                    if now - last_feedback >= feedback_interval:
                        last_feedback = now
                        sm_names = {
                            1: "NORMAL", 2: "REDUCED",
                            3: "PROTECTIVE_STOP", 4: "RECOVERY",
                            5: "SAFEGUARD_STOP",
                            6: "SYS_ESTOP", 7: "ROBOT_ESTOP",
                            8: "VIOLATION", 9: "FAULT",
                            12: "AUTO_SAFEGUARD",
                        }
                        sm_str = sm_names.get(
                            sm, f"UNKNOWN({sm})"
                        )
                        self._log.info(
                            f"Recovery [{robot_name}]: mode=7 "
                            f"but safety_mode={sm_str} — "
                            f"still waiting (attempt {attempt})"
                        )
                    continue  # keep polling

            # Not yet recovered — send periodic updates
            now = time.monotonic()
            if now - last_feedback >= feedback_interval:
                last_feedback = now
                mode_names = {
                    0: "DISCONNECTED", 1: "CONFIRM_SAFETY",
                    2: "BOOTING", 3: "POWER_OFF",
                    4: "POWER_ON", 5: "IDLE",
                    6: "BACKDRIVE", 7: "RUNNING",
                }
                mode_str = mode_names.get(robot_mode, f"UNKNOWN({robot_mode})")
                self._log.info(
                    f"Recovery [{robot_name}]: waiting… "
                    f"robot_mode={mode_str} (attempt {attempt})"
                )
                await self._send_feedback(
                    client, request_id, pose_idx, total,
                    (pose_idx / max(total, 1)) * 100,
                    "fault_paused",
                    message=(
                        f"⏳ Waiting for {robot_name} to recover: "
                        f"mode={mode_str}. Reset the robot, start "
                        f"the program, and put in remote control. "
                        f"(attempt {attempt})"
                    ),
                )

        return False  # stop_requested

    async def _home_if_needed(self, robot_name, cfg, mode):
        """Move the robot home if needed.

        Returns
        -------
        (ok, error_msg) : tuple[bool, str]
            ok=True if the robot is at home (or was moved there).
            ok=False + error_msg if home move failed.
        """
        home = cfg["home_position"]
        st = self._js_mgr.robot_states[robot_name]
        current = st.joint_positions

        at_home = (
            current and len(current) == 6
            and all(abs(current[i] - home[i]) < 0.05 for i in range(6))
        )

        if (mode in ("real", "both") and RTDE_AVAILABLE
                and robot_name in self._executor._rtde):
            rtde = self._executor._rtde[robot_name]
            if not rtde.is_connected:
                rtde.connect()
            if rtde.is_connected:
                actual = rtde.get_actual_q()
                if actual:
                    at_home = all(
                        abs(actual[i] - home[i]) < 0.05 for i in range(6)
                    )

        if not at_home:
            self._log.info(f"Moving {robot_name} to HOME…")
            ok = await self._executor.execute_home(robot_name)
            if not ok:
                # Check for a descriptive error from the RTDE controller
                err = ""
                if (RTDE_AVAILABLE
                        and robot_name in self._executor._rtde):
                    err = self._executor._rtde[robot_name].last_error
                return (False, err or "Home move failed (unknown reason)")
        return (True, "")

    def _detect_hardware_issues(self):
        """Detect hardware faults on all RTDE-controlled robots.

        Returns a list of issues.  Safeguard stops are reported but
        are NOT treated as fatal — the caller decides whether to
        pause or abort.

        Uses ``get_safety_mode()`` for ALL checks because it does
        NOT gate on ``_recv_healthy`` — during and after servoJ the
        recv interface is alive but healthy-flag is False, so the
        per-method checks (isProtectiveStopped, isEmergencyStopped)
        would silently return False and miss real faults.

        Does NOT use ``last_error`` — that reflects RTDE API state
        (e.g. previous _ensure_ctrl failure), not the robot's live
        safety state. Using it caused false-positive recovery loops
        on every IK-failed pose.
        """
        issues = []
        mode = getattr(self._node, '_current_mode', 'simulation')
        if mode in ("real", "both"):
            for rn, rtde in self._executor._rtde.items():
                sm = rtde.get_safety_mode()
                if sm in (
                    rtde.SAFETY_MODE_SAFEGUARD_STOP,
                    rtde.SAFETY_MODE_AUTO_SAFEGUARD_STOP,
                ):
                    issues.append(f"{rn}:SAFEGUARD_STOP")
                elif sm == rtde.SAFETY_MODE_PROTECTIVE_STOP:
                    issues.append(f"{rn}:PROTECTIVE_STOP")
                elif sm in (
                    rtde.SAFETY_MODE_SYSTEM_EMERGENCY_STOP,
                    rtde.SAFETY_MODE_ROBOT_EMERGENCY_STOP,
                ):
                    issues.append(f"{rn}:EMERGENCY_STOP")
                elif sm == rtde.SAFETY_MODE_VIOLATION:
                    issues.append(f"{rn}:SAFETY_VIOLATION")
                elif sm == rtde.SAFETY_MODE_FAULT:
                    issues.append(f"{rn}:SAFETY_FAULT")
                # sm == NORMAL/REDUCED/-1/RECOVERY → no issue
        return issues

    def _has_fatal_hardware_issue(self, issues: list) -> bool:
        """Return True if any issue is NOT a recoverable safeguard stop."""
        return any(
            not issue.endswith(":SAFEGUARD_STOP") for issue in issues
        )

    async def _send_feedback(
        self, client, request_id, pose_idx, total, progress, status,
        message=None, current_pose_name=None,
    ):
        payload = {
            "type": "rpc_feedback",
            "request_id": request_id,
            "current_pose_index": pose_idx,
            "total_poses": total,
            "progress_percent": progress,
            "status": status,
        }
        if message:
            payload["message"] = message
        if current_pose_name:
            payload["current_pose_name"] = current_pose_name
        await client.websocket.send(json.dumps(payload))

    async def _send_error(self, client, rid, error):
        try:
            await client.websocket.send(json.dumps({
                "type": "error", "request_id": rid, "error": error,
            }))
        except Exception:
            pass
