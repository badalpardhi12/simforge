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
        finally:
            self.running = False

    # ── Inner implementation ─────────────────────────────────────

    async def _run_inner(
        self, client, request_id, robot_name, poses,
        idle_time, mode, move_speed, cfg, total,
        go_home_before, go_home_after,
    ):
        self._log.info(
            f"Proto-sim START (cuRobo): {total} poses on {robot_name} "
            f"(mode={mode}, speed={move_speed}, idle={idle_time}s, "
            f"home_before={go_home_before}, home_after={go_home_after})"
        )

        # ── Optional home-before ─────────────────────────────────
        if go_home_before:
            await self._home_if_needed(robot_name, cfg, mode)

        # ── Plan ─────────────────────────────────────────────────
        await self._send_feedback(client, request_id, 0, total, 0,
                                  "planning",
                                  f"Planning trajectory through {total} "
                                  f"poses (cuRobo GPU)…")

        current = self._js_mgr.robot_states[robot_name].joint_positions
        if not current or len(current) != 6:
            current = list(cfg["home_position"])

        plan_result = await self._planner.plan_multi_waypoint(
            robot_name, poses,
            current_joints=current,
            velocity_scaling=move_speed,
            idle_time=idle_time,
        )

        if plan_result is None:
            self._log.warn(
                "Multi-waypoint planning failed — fallback"
            )
            await self._run_legacy(
                client, request_id, robot_name, poses,
                idle_time, mode, move_speed, cfg, total,
            )
            return

        trajectory, pose_times, valid_indices = plan_result
        ik_failed = total - len(valid_indices)

        pts = trajectory.joint_trajectory.points
        total_dur = (
            pts[-1].time_from_start.sec
            + pts[-1].time_from_start.nanosec * 1e-9
        ) if pts else 0.0

        self._log.info(
            f"Executing unified trajectory: {len(pts)} points, "
            f"duration={total_dur:.2f}s"
        )

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

        # ── Execute ──────────────────────────────────────────────
        exec_timeout = max(total_dur * 2.0, 60.0)
        exec_task = asyncio.ensure_future(
            self._executor.execute(
                robot_name, trajectory, timeout=exec_timeout
            )
        )

        exec_start = time.monotonic()
        pose_time_idx = 0
        last_reported = -1

        while not exec_task.done():
            elapsed = time.monotonic() - exec_start
            progress = min(elapsed / total_dur, 1.0) if total_dur > 0 else 1.0

            while (pose_time_idx < len(pose_times)
                   and elapsed >= pose_times[pose_time_idx][1]):
                pi = pose_times[pose_time_idx][0]
                if pi != last_reported:
                    pname = poses[pi].get("name", f"pose_{pi}")
                    await self._send_feedback(
                        client, request_id, pi, total,
                        progress * 100, "reached",
                        current_pose_name=pname,
                    )
                    self._log.info(
                        f"  [{pi+1}/{total}] ✓ passed {pname}"
                    )
                    last_reported = pi
                pose_time_idx += 1

            if self.stop_requested:
                self._log.info("Proto-sim stop requested")
                break
            await asyncio.sleep(0.5)

        try:
            exec_ok = await asyncio.wait_for(
                asyncio.shield(exec_task), timeout=10.0
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            exec_ok = False

        completed = len(valid_indices) if exec_ok else 0
        plan_failed = 0 if exec_ok else 1

        # ── Optional home-after ──────────────────────────────────
        if go_home_after and completed > 0 and not self.stop_requested:
            self._log.info(f"Returning {robot_name} to home…")
            await self._executor.execute_home(robot_name)

        self.running = False

        # ── Build result ─────────────────────────────────────────
        summary_parts = [f"Completed {completed}/{total} poses (cuRobo)"]
        if ik_failed:
            summary_parts.append(f"{ik_failed} IK failed")
        if plan_failed:
            summary_parts.append("trajectory execution failed")

        hardware_issues = self._detect_hardware_issues()
        if hardware_issues:
            summary_parts.append(
                f"⚠ Hardware: {', '.join(hardware_issues)} disconnected"
            )

        summary = " | ".join(summary_parts)
        self._log.info(f"Proto-sim DONE: {summary}")

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

    # ── Legacy fallback (pose-by-pose) ───────────────────────────

    async def _run_legacy(
        self, client, request_id, robot_name, poses,
        idle_time, mode, move_speed, cfg, total,
    ):
        """Pose-by-pose cuRobo fallback."""
        completed, ik_failed, plan_failed = 0, 0, 0
        self._log.info("Running pose-by-pose execution (cuRobo fallback)")

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
        summary = f"Completed {completed}/{total} poses (cuRobo legacy)"
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
            await self._executor.execute_home(robot_name)

    def _detect_hardware_issues(self):
        issues = []
        mode = getattr(self._node, '_current_mode', 'simulation')
        if mode in ("real", "both"):
            prog = getattr(self._node, '_robot_program_running', {})
            for rn in ROBOT_CONFIG:
                if not prog.get(rn, False):
                    issues.append(rn)
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
