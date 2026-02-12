"""
RPC Handlers — one method per WebSocket RPC.

Each handler is independently testable.
"""

import asyncio
import json
import os
import time
import traceback
from typing import Any, Dict, Optional

import rclpy

from .config import (
    ROBOT_CONFIG, KNOWN_OBJECTS,
    RTDE_AVAILABLE, CUROBO_AVAILABLE,
    MODE_SWITCH_FILE, STACK_READY_FILE,
    RobotStateInfo,
)


class RPCHandlers:
    """Collection of RPC method implementations.

    Parameters
    ----------
    node : CommandGatewayNode
        The ROS2 node for TF, logging, parameters, etc.
    planner : CuroboPlanner
        GPU motion planner.
    executor : TrajectoryExecutor
        Trajectory dispatch (RTDE / ROS2 / dual).
    proto_exec : ProtocolExecutor
        Multi-pose protocol runner.
    js_mgr : JointStateManager
        Joint state publisher / broadcaster control.
    """

    def __init__(self, node, planner, executor, proto_exec, js_mgr):
        self._node = node
        self._log = node.get_logger()
        self._planner = planner
        self._executor = executor
        self._proto = proto_exec
        self._js_mgr = js_mgr

    # ── Dispatch ─────────────────────────────────────────────────

    async def dispatch(self, client, msg: dict):
        """Route an RPC message to the correct handler."""
        rid = msg.get("request_id")
        method = msg.get("method", "")
        params = msg.get("params", {})
        self._log.info(f"RPC from {client.client_id}: {method}")

        try:
            if method == "get_environment_info":
                result = await self.env_info(params)
            elif method == "get_robot_status":
                result = await self.robot_status(params)
            elif method == "prepare_mode":
                result = await self.prepare_mode(params)
            elif method == "run_proto_sim":
                asyncio.ensure_future(
                    self.run_proto_sim(client, rid, params)
                )
                return  # result sent asynchronously
            elif method == "stop_proto_sim":
                self._proto.stop_requested = True
                result = {"success": True, "message": "Stop requested"}
            elif method == "move_home":
                result = await self.move_home(params)
            elif method == "check_collision":
                result = {"success": True, "in_collision": False}
            elif method == "plan_motion":
                result = await self.plan_motion(params)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown RPC method: {method}",
                }

            await client.websocket.send(json.dumps({
                "type": "rpc_result", "request_id": rid, **result,
            }))
        except Exception as e:
            self._log.error(
                f"RPC error ({method}): {e}\n{traceback.format_exc()}"
            )
            await self._send_error(client, rid, str(e))

    # ── get_environment_info ─────────────────────────────────────

    async def env_info(self, params) -> dict:
        default_robot = list(ROBOT_CONFIG.keys())[0]
        ref = ROBOT_CONFIG[default_robot]["base_link"]
        obj_tf = {}
        for obj in KNOWN_OBJECTS:
            try:
                t = self._node.tf_buffer.lookup_transform(
                    ref, obj, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=2.0),
                )
                obj_tf[obj] = {
                    "position": [
                        t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z,
                    ],
                    "orientation": [
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w,
                    ],
                }
            except Exception as e:
                self._log.warn(f"TF for {obj}: {e}")
                obj_tf[obj] = None
        return {
            "success": True,
            "robots": list(ROBOT_CONFIG.keys()),
            "objects": KNOWN_OBJECTS,
            "object_transforms": obj_tf,
            "reference_frame": ref,
        }

    # ── get_robot_status ─────────────────────────────────────────

    async def robot_status(self, params) -> dict:
        rn = params.get("robot_name") or list(ROBOT_CONFIG.keys())[0]
        cfg = ROBOT_CONFIG.get(rn)
        if not cfg:
            return {"success": False, "error": f"Unknown robot: {rn}"}

        st = self._js_mgr.robot_states.get(rn, RobotStateInfo())
        has_joints = len(st.joint_positions) == 6

        traj_ok = (rn in self._executor._traj_clients
                   and self._executor._traj_clients[rn].server_is_ready())
        curobo_ok = self._planner.has_robot(rn)

        rtde_ok = False
        if RTDE_AVAILABLE and rn in self._executor._rtde:
            rtde_ok = self._executor._rtde[rn].is_connected

        current_mode = self._node._current_mode
        is_real = current_mode in ("real", "both")
        prog_running = self._node._robot_program_running.get(rn, False)

        return {
            "success": True,
            "real_robot_available": is_real and (rtde_ok or traj_ok) and has_joints,
            "simulation_available": True,
            "available_modes": ["simulation", "real"],
            "current_mode": current_mode,
            "connection_details": {
                "follow_trajectory_action": (
                    "available" if traj_ok else "not_available"
                ),
                "curobo": "available" if curobo_ok else "not_available",
                "rtde": (
                    "connected" if rtde_ok
                    else ("available" if RTDE_AVAILABLE else "not_installed")
                ),
                "robot_program_running": prog_running,
            },
            "current_joint_positions": (
                st.joint_positions if has_joints else list(cfg["home_position"])
            ),
            "position_source": "real_robot" if (is_real and has_joints) else "default",
        }

    # ── prepare_mode ─────────────────────────────────────────────

    async def prepare_mode(self, params) -> dict:
        mode = params.get("mode", "simulation")
        self._log.info(f"Preparing mode: {mode}")

        if self._proto.running:
            return {
                "success": False, "ready": False,
                "message": "Cannot switch mode while a protocol is running",
                "can_retry": True,
            }

        if mode in ("real", "both"):
            if not RTDE_AVAILABLE:
                return {
                    "success": False, "ready": False,
                    "message": "ur_rtde not installed — cannot use real mode",
                    "can_retry": False,
                }

            # Connect RTDE controllers
            connected, failed = [], []
            for rn, rtde in self._executor._rtde.items():
                if rtde.is_connected:
                    connected.append(rn)
                    continue
                self._log.info(f"Connecting RTDE to {rn}...")
                if rtde.connect():
                    connected.append(rn)
                else:
                    failed.append(rn)

            if failed:
                return {
                    "success": True, "ready": False,
                    "message": f"RTDE connected: {connected}, failed: {failed}",
                    "can_retry": True,
                }

            # Deactivate sim broadcaster FIRST so there is no window
            # where both the broadcaster (~500 Hz) and the RTDE
            # publisher (~50 Hz) are active simultaneously.
            self._js_mgr.set_sim_broadcaster(active=False)

            # Now start RTDE → /joint_states publisher
            self._js_mgr.start_rtde_publisher(
                self._executor._rtde
            )

            self._node._current_mode = mode
            self._log.info(f"Real mode ready — RTDE connected to {connected}")
            return {
                "success": True, "ready": True,
                "message": (
                    f"Real mode ready — RTDE direct control on "
                    f"{connected} (ROS2 sim stack kept for Foxglove)"
                ),
                "can_retry": False,
            }

        else:
            # Simulation mode — disconnect RTDE
            for rn, rtde in self._executor._rtde.items():
                if rtde.is_connected:
                    rtde.disconnect()
            self._js_mgr.stop_rtde_publisher()
            self._js_mgr.set_sim_broadcaster(active=True)

            self._node._current_mode = "simulation"
            return {
                "success": True, "ready": True,
                "message": "Simulation mode ready (cuRobo backend)",
                "can_retry": False,
            }

    # ── run_proto_sim ────────────────────────────────────────────

    async def run_proto_sim(self, client, request_id, params):
        robot_name = params.get("robot_name", list(ROBOT_CONFIG.keys())[0])
        poses = params.get("poses", [])
        idle_time = params.get("idle_time", 2.0)
        mode = params.get("mode", "simulation")
        requested_speed = params.get(
            "move_speed", self._node.max_velocity_scaling
        )
        move_speed = min(self._node.max_velocity_scaling, requested_speed)

        if robot_name not in ROBOT_CONFIG:
            await self._send_error(
                client, request_id, f"Unknown robot: {robot_name}"
            )
            return

        if not self._planner.has_robot(robot_name):
            await self._send_error(
                client, request_id,
                f"cuRobo not initialised for {robot_name}",
            )
            return

        go_home_before = params.get("go_home_before", False)
        go_home_after = params.get("go_home_after", False)

        await self._proto.run(
            client, request_id, robot_name, poses,
            idle_time=idle_time,
            mode=mode,
            move_speed=move_speed,
            go_home_before=go_home_before,
            go_home_after=go_home_after,
        )

    # ── move_home ────────────────────────────────────────────────

    async def move_home(self, params) -> dict:
        robot_name = params.get("robot_name", list(ROBOT_CONFIG.keys())[0])
        if robot_name not in ROBOT_CONFIG:
            return {"success": False, "message": f"Unknown robot: {robot_name}"}

        self._log.info(f"RPC move_home for {robot_name}")
        try:
            ok = await self._executor.execute_home(robot_name)
            if ok:
                return {"success": True, "message": f"{robot_name} at home position"}
            else:
                return {"success": False, "message": f"Failed to move {robot_name} home"}
        except Exception as e:
            self._log.error(f"move_home error: {e}")
            return {"success": False, "message": str(e)}

    # ── plan_motion ──────────────────────────────────────────────

    async def plan_motion(self, params) -> dict:
        rn = params.get("robot_name", list(ROBOT_CONFIG.keys())[0])
        target = params.get("target_joints")
        if not target or len(target) != 6:
            return {"success": False, "error": "target_joints with 6 values required"}

        current = self._js_mgr.robot_states[rn].joint_positions
        if not current or len(current) != 6:
            current = list(ROBOT_CONFIG[rn]["home_position"])

        result = await self._planner.plan_to_joints(
            rn, target, current_joints=current,
        )
        if result is None:
            return {"success": False, "error": "Planning failed"}

        traj = result.get_interpolated_plan()
        return {"success": True, "waypoints": traj.position.shape[1]}

    # ── Mode switch wait helper ──────────────────────────────────

    async def wait_for_ros_stack(
        self, target_mode: str, timeout: float = 120.0, progress_cb=None,
    ) -> dict:
        """Wait for the ROS2 stack to come back up after a mode switch."""
        switch_start = time.monotonic()
        overall_deadline = switch_start + timeout

        async def _report(phase: str, detail: str):
            msg = f"[{phase}] {detail}"
            self._log.info(msg)
            if progress_cb:
                try:
                    await progress_cb(phase, detail)
                except Exception:
                    pass

        # Phase 1: supervisor picks up signal
        await _report("Phase 1/5", "Waiting for supervisor…")
        phase1_deadline = min(switch_start + 30.0, overall_deadline)
        while time.monotonic() < phase1_deadline:
            if (not os.path.exists(MODE_SWITCH_FILE)
                    and not os.path.exists(STACK_READY_FILE)):
                break
            await asyncio.sleep(0.5)

        # Phase 2: STACK_READY_FILE
        await _report("Phase 2/5", "Waiting for ROS2 stack…")
        phase2_deadline = min(time.monotonic() + 90.0, overall_deadline)
        stack_ready = False
        while time.monotonic() < phase2_deadline:
            if os.path.exists(STACK_READY_FILE):
                stack_ready = True
                break
            await asyncio.sleep(1.0)
        if not stack_ready:
            return {"ok": False, "phase": "Phase 2/5",
                    "message": "Timed out waiting for ROS2 stack"}

        # Phase 3: mode verification
        await _report("Phase 3/5", "Verifying mode…")
        cur = self._node._current_mode
        target_is_real = target_mode in ("real", "both")
        cur_is_real = cur in ("real", "both")
        if target_is_real != cur_is_real:
            return {"ok": False, "phase": "Phase 3/5",
                    "message": f"Mode mismatch: wanted '{target_mode}', got '{cur}'"}

        # Phase 4: fresh joint states
        await _report("Phase 4/5", "Waiting for fresh joint states…")
        phase4_deadline = min(time.monotonic() + 20.0, overall_deadline)
        while time.monotonic() < phase4_deadline:
            all_fresh = all(
                self._js_mgr.robot_states.get(rn)
                and self._js_mgr.robot_states[rn].last_update > switch_start
                and len(self._js_mgr.robot_states[rn].joint_positions) == 6
                for rn in ROBOT_CONFIG
            )
            if all_fresh:
                break
            await asyncio.sleep(0.5)

        # Phase 5: trajectory servers
        await _report("Phase 5/5", "Waiting for trajectory servers…")
        phase5_deadline = min(time.monotonic() + 15.0, overall_deadline)
        while time.monotonic() < phase5_deadline:
            all_ok = all(
                tc.server_is_ready()
                for tc in self._executor._traj_clients.values()
            )
            if all_ok:
                break
            await asyncio.sleep(1.0)
        else:
            not_ready = [
                rn for rn, tc in self._executor._traj_clients.items()
                if not tc.server_is_ready()
            ]
            return {"ok": False, "phase": "Phase 5/5",
                    "message": f"Trajectory servers not ready: {not_ready}"}

        # Real mode: wait for robot programs
        if self._node._current_mode in ("real", "both"):
            await _report("Phase 5/5", "Waiting for robot programs…")
            phase6_deadline = min(time.monotonic() + 30.0, overall_deadline)
            while time.monotonic() < phase6_deadline:
                if all(
                    self._node._robot_program_running.get(rn, False)
                    for rn in ROBOT_CONFIG
                ):
                    break
                await asyncio.sleep(0.5)

        elapsed = time.monotonic() - switch_start
        return {"ok": True, "phase": "Complete",
                "message": f"ROS2 stack ready in {elapsed:.1f}s"}

    # ── Utility ──────────────────────────────────────────────────

    async def _send_error(self, client, rid, error):
        try:
            await client.websocket.send(json.dumps({
                "type": "error", "request_id": rid, "error": error,
            }))
        except Exception:
            pass
