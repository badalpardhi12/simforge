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
import time
import traceback
from typing import Optional

import websockets

from .config import ROBOT_CONFIG, RTDE_AVAILABLE, RobotStateInfo


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
            f"Proto-sim START (cuRobo pipelined): {total} poses on "
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
            if rtde is not None and rtde.last_error:
                err = rtde.last_error
                self._log.error(
                    f"RTDE pre-flight failed for {robot_name}: {err}"
                )
                await self._send_error(
                    client, request_id,
                    f"Robot {robot_name} is not controllable: {err}. "
                    f"Ensure the robot program is running on the "
                    f"teach pendant."
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

        # ── Pipelined plan-execute ───────────────────────────────
        # Producer plans segments via cuRobo (CUDA in background thread)
        # Consumer executes planned segments via ROS2/RTDE.
        # Because plan_to_pose now uses asyncio.to_thread(), the event
        # loop stays free so the consumer can drive the robot while the
        # producer plans the next segment.

        segment_queue = asyncio.Queue()  # unbounded — avoids deadlocks
        hw_abort = False
        completed = 0
        ik_failed = 0
        exec_failed = 0

        current_joints = (
            self._js_mgr.robot_states[robot_name].joint_positions
        )
        if not current_joints or len(current_joints) != 6:
            current_joints = list(cfg["home_position"])

        await self._send_feedback(
            client, request_id, 0, total, 0, "planning",
            f"Starting pipelined plan-execute for {total} poses "
            f"(cuRobo GPU)…",
        )

        async def producer():
            """Plan each segment; push to queue for the consumer."""
            joints = list(current_joints)
            try:
                for i, pose_data in enumerate(poses):
                    if self.stop_requested:
                        break
                    pose_name = pose_data.get("name", f"pose_{i}")
                    position = pose_data.get("position", [0, 0, 0])
                    orientation = pose_data.get(
                        "orientation", [0, 0, 0, 1]
                    )

                    pct = (i / total) * 50.0
                    await self._send_feedback(
                        client, request_id, i, total, pct,
                        "planning",
                        f"Planning {i+1}/{total}: {pose_name}",
                    )

                    t0 = time.monotonic()
                    try:
                        result = await self._planner.plan_to_pose(
                            robot_name, position, orientation,
                            current_joints=joints,
                            velocity_scaling=move_speed,
                        )
                    except Exception as e:
                        self._log.error(
                            f"Planning exception for {pose_name}: {e}"
                        )
                        result = None
                    dt = time.monotonic() - t0

                    if result is None:
                        self._log.warn(
                            f"  [{i+1}/{total}] ✗ plan failed "
                            f"{pose_name} ({dt:.1f}s)"
                        )
                        await segment_queue.put({
                            "action": "skip",
                            "index": i,
                            "pose_name": pose_name,
                        })
                        continue

                    ros_traj = self._planner.result_to_ros_trajectory(
                        result, robot_name,
                    )

                    # Final joints for the next segment's start state
                    last_pt = ros_traj.joint_trajectory.points[-1]
                    joints = list(last_pt.positions)

                    self._log.info(
                        f"  [{i+1}/{total}] planned "
                        f"{pose_name} ({dt:.1f}s)"
                    )
                    await segment_queue.put({
                        "action": "execute",
                        "index": i,
                        "pose_name": pose_name,
                        "trajectory": ros_traj,
                    })
            except Exception as e:
                self._log.error(
                    f"Producer error: {e}\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                # Sentinel — tells the consumer no more segments
                await segment_queue.put(None)

        async def consumer():
            """Execute planned segments as they arrive."""
            nonlocal completed, ik_failed, exec_failed, hw_abort

            while True:
                item = await segment_queue.get()
                if item is None:
                    break
                if self.stop_requested:
                    break

                if item["action"] == "skip":
                    ik_failed += 1
                    continue

                i = item["index"]
                pose_name = item["pose_name"]
                ros_traj = item["trajectory"]

                # Hardware fault check
                hw_issues = self._detect_hardware_issues()
                if hw_issues:
                    hw_abort = True
                    self._log.error(
                        f"Hardware fault: {', '.join(hw_issues)} "
                        f"— aborting"
                    )
                    await self._send_feedback(
                        client, request_id, i, total,
                        50 + (completed / max(total, 1)) * 50,
                        "error",
                        message=(
                            f"⚠ HARDWARE FAULT: "
                            f"{', '.join(hw_issues)}. "
                            f"Robot stopped — aborting protocol."
                        ),
                    )
                    self.stop_requested = True
                    break

                pct = 50 + (completed / max(total, 1)) * 50
                await self._send_feedback(
                    client, request_id, i, total, pct,
                    "moving", current_pose_name=pose_name,
                )

                # Compute per-segment execution timeout
                pts = ros_traj.joint_trajectory.points
                traj_dur = (
                    pts[-1].time_from_start.sec
                    + pts[-1].time_from_start.nanosec * 1e-9
                ) if pts else 5.0
                exec_timeout = max(traj_dur * 2.0, 30.0)

                ok = await self._executor.execute(
                    robot_name, ros_traj, timeout=exec_timeout,
                )

                if ok:
                    completed += 1
                    self._log.info(
                        f"  [{i+1}/{total}] ✓ {pose_name}"
                    )
                    await self._send_feedback(
                        client, request_id, i, total,
                        50 + (completed / max(total, 1)) * 50,
                        "reached", current_pose_name=pose_name,
                    )
                    if idle_time > 0 and i < len(poses) - 1:
                        await asyncio.sleep(idle_time)
                else:
                    exec_failed += 1
                    self._log.error(
                        f"  [{i+1}/{total}] ✗ exec failed {pose_name}"
                    )

        # Run producer and consumer concurrently.
        # plan_to_pose uses asyncio.to_thread() so CUDA planning
        # runs in a background thread — the consumer can execute
        # trajectories on the event loop at the same time.
        await asyncio.gather(producer(), consumer())

        # ── Optional home-after ──────────────────────────────────
        if go_home_after and completed > 0 and not self.stop_requested:
            self._log.info(f"Returning {robot_name} to home…")
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
            f"Completed {completed}/{total} poses (cuRobo pipelined)"
        ]
        if ik_failed:
            summary_parts.append(f"{ik_failed} IK/plan failed")
        if exec_failed:
            summary_parts.append(f"{exec_failed} execution failed")

        hardware_issues = self._detect_hardware_issues()
        if hardware_issues:
            summary_parts.append(
                f"⚠ Hardware: {', '.join(hardware_issues)}"
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
        issues = []
        mode = getattr(self._node, '_current_mode', 'simulation')
        if mode in ("real", "both"):
            # In RTDE mode we bypass the ROS2 UR driver, so
            # _robot_program_running is irrelevant.  Check RTDE
            # controllers directly for safety faults.
            for rn, rtde in self._executor._rtde.items():
                if rtde.is_protective_stopped():
                    issues.append(f"{rn}:PROTECTIVE_STOP")
                elif rtde.is_emergency_stopped():
                    issues.append(f"{rn}:EMERGENCY_STOP")
                elif rtde.last_error:
                    # Truncate the error to fit in the JSON result
                    issues.append(f"{rn}:{rtde.last_error[:80]}")
        return issues

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
