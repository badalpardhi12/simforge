"""
Trajectory Executor — dispatches trajectories to RTDE and/or ROS2.

Handles:
  • RTDE servoJ streaming (real robot)
  • ROS2 FollowJointTrajectory action (simulation)
  • Dual dispatch: send to RTDE *and* sim in parallel so Foxglove matches
"""

import asyncio
import time
from typing import Dict, Optional

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

from .config import ROBOT_CONFIG, RTDE_AVAILABLE


# ── Async ROS2 future helper ────────────────────────────────────


async def await_ros_future(future, timeout: float = 10.0):
    """Poll an rclpy Future from an asyncio coroutine."""
    deadline = time.monotonic() + timeout
    while not future.done():
        if time.monotonic() > deadline:
            raise TimeoutError(f"ROS future timed out after {timeout}s")
        await asyncio.sleep(0.05)
    return future.result()


class TrajectoryExecutor:
    """Dispatches trajectories to the appropriate execution backend."""

    def __init__(self, node, *, rtde_controllers: dict = None,
                 stop_check_fn=None, joint_state_manager=None):
        self._node = node
        self._log = node.get_logger()
        self._rtde = rtde_controllers or {}
        self._stop_check_fn = stop_check_fn
        self._js_mgr = joint_state_manager  # for position publishing

        # FollowJointTrajectory action clients (sim)
        self._traj_clients: Dict[str, ActionClient] = {}
        self._rebuild_traj_clients()

    # ── Mode query ───────────────────────────────────────────────

    def _current_mode(self) -> str:
        return getattr(self._node, '_current_mode', 'simulation')

    # ── Public API ───────────────────────────────────────────────

    async def execute(
        self, robot_name: str, trajectory, timeout: float = 60.0,
    ) -> bool:
        """Execute a trajectory — dispatches to RTDE and/or ROS2."""
        mode = self._current_mode()
        is_real = mode in ("real", "both")

        if is_real and RTDE_AVAILABLE and robot_name in self._rtde:
            return await self._execute_rtde(robot_name, trajectory, timeout)
        else:
            return await self._execute_ros2(robot_name, trajectory, timeout)

    async def execute_home(
        self, robot_name: str, velocity_scaling: float = 0.5,
    ) -> bool:
        """Move a robot to its home position.

        Real mode → RTDE moveJ.  Sim mode → cuRobo plan + ROS2 execute.
        """
        cfg = ROBOT_CONFIG[robot_name]
        mode = self._current_mode()
        is_real = mode in ("real", "both")

        if is_real and RTDE_AVAILABLE and robot_name in self._rtde:
            rtde = self._rtde[robot_name]
            if not rtde.is_connected:
                rtde.connect()
            if rtde.is_connected:
                self._log.info(
                    f"Moving {robot_name} home via RTDE moveJ"
                )
                loop = asyncio.get_event_loop()
                ok = await loop.run_in_executor(
                    None,
                    lambda: rtde.move_j(
                        cfg["home_position"], speed=0.5, acceleration=0.5
                    ),
                )
                # moveJ can leave the receive interface dead —
                # reconnect so the 50 Hz timer can read positions.
                # The recv can die again ~2 s after moveJ, so we
                # do a short wait + verify loop (max 3 attempts).
                for attempt in range(3):
                    await loop.run_in_executor(
                        None, rtde.reconnect_receive,
                    )
                    if rtde._recv_healthy:
                        # Wait a moment and re-verify
                        await asyncio.sleep(1.0)
                        test_q = await loop.run_in_executor(
                            None, rtde.get_actual_q,
                        )
                        if test_q is not None:
                            break
                        self._log.info(
                            f"RTDE recv for {robot_name} died again "
                            f"(attempt {attempt + 1}/3), retrying..."
                        )
                return ok
        # Sim fallback handled by caller (needs planner)
        return False

    # ── Action client management ─────────────────────────────────

    def rebuild_traj_clients(self):
        self._rebuild_traj_clients()

    def _rebuild_traj_clients(self) -> None:
        for old in self._traj_clients.values():
            old.destroy()
        self._traj_clients.clear()
        for rn, cfg in ROBOT_CONFIG.items():
            action_name = (
                f"/{cfg['controller']}/follow_joint_trajectory"
            )
            self._traj_clients[rn] = ActionClient(
                self._node, FollowJointTrajectory, action_name,
            )
            self._log.info(
                f"Trajectory action client for {rn}: {action_name}"
            )

    @property
    def traj_clients(self):
        return self._traj_clients

    # ── RTDE execution ───────────────────────────────────────────

    async def _execute_rtde(
        self, robot_name: str, trajectory, timeout: float = 60.0,
    ) -> bool:
        """Stream trajectory via RTDE servoJ, and also send to sim
        in parallel so the Foxglove model stays in sync."""
        rtde = self._rtde.get(robot_name)
        if rtde is None:
            self._log.error(f"No RTDE controller for {robot_name}")
            return False

        if not rtde.is_connected:
            if not rtde.connect():
                self._log.error(
                    f"RTDE connection failed for {robot_name}"
                )
                return False

        pts = trajectory.joint_trajectory.points
        if not pts:
            self._log.warn(f"Empty trajectory for {robot_name}")
            return True

        positions = [list(pt.positions) for pt in pts]
        timestamps = [
            pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            for pt in pts
        ]
        total_dur = timestamps[-1] if timestamps else 0.0

        self._log.info(
            f"Executing via RTDE servoJ on {robot_name}: "
            f"{len(positions)} waypoints, duration={total_dur:.2f}s "
            f"(timestamps used for dwell support)"
        )

        # In real mode, do NOT fire the sim trajectory.
        # The position_callback at ~50 Hz publishes commanded positions
        # directly to /joint_states, which is the sole source of truth.
        # Sending the trajectory to the sim's FollowJointTrajectory causes
        # the active scaled_joint_trajectory_controller to write positions
        # into the fake hardware.  Even with joint_state_broadcaster
        # deactivated, the controller's execution on fake hardware can
        # cause /joint_states to jump between the sim-interpolated
        # trajectory (starting from the old home position) and the real
        # RTDE positions — producing the glitchy jumping effect.

        # Build position callback for real-time /joint_states
        pos_cb = None
        if self._js_mgr is not None:
            def pos_cb(rn, q):
                self._js_mgr.publish_positions(rn, q)

        # Blocking servoJ in executor thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rtde.execute_trajectory_servoj(
                positions=positions,
                timestamps=timestamps,
                lookahead_time=0.1,
                gain=300,
                stop_check_fn=self._stop_check_fn,
                logger=self._log,
                position_callback=pos_cb,
            ),
        )

        return result

    def _start_sim_trajectory(self, robot_name, trajectory, total_dur):
        """Optionally send the trajectory to the sim FollowJointTrajectory."""
        try:
            sim_client = self._traj_clients.get(robot_name)
            if sim_client is not None and sim_client.server_is_ready():
                self._log.info(
                    f"Sending trajectory to sim for {robot_name} in parallel"
                )
                return asyncio.ensure_future(
                    self._execute_ros2(
                        robot_name, trajectory,
                        timeout=max(total_dur * 2, 60.0),
                    )
                )
            self._log.info(
                "Sim trajectory action not available — "
                "sim robot will not mirror real movement"
            )
        except Exception as e:
            self._log.warn(f"Could not start sim trajectory: {e}")
        return None

    # ── ROS2 FollowJointTrajectory execution ─────────────────────

    async def _execute_ros2(
        self, robot_name: str, trajectory, timeout: float = 60.0,
    ) -> bool:
        """Send trajectory via FollowJointTrajectory action (simulation)."""
        client = self._traj_clients.get(robot_name)
        if client is None:
            self._log.error(f"No trajectory client for {robot_name}")
            return False

        max_attempts = 2
        for attempt in range(max_attempts):
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if client.server_is_ready():
                    break
                await asyncio.sleep(0.2)
            else:
                self._log.error(
                    f"Trajectory action not available for {robot_name}"
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(5.0)
                    continue
                return False

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = trajectory.joint_trajectory
            goal.goal_time_tolerance.sec = 30

            pts = goal.trajectory.points
            # Strip zero-time leading point
            if len(pts) >= 2:
                t0 = (pts[0].time_from_start.sec
                      + pts[0].time_from_start.nanosec * 1e-9)
                t1 = (pts[1].time_from_start.sec
                      + pts[1].time_from_start.nanosec * 1e-9)
                if t0 >= t1 or t0 == 0.0:
                    pts = list(pts[1:])
                    goal.trajectory.points = pts

            # Verify monotonic
            if len(pts) >= 2:
                times = [
                    p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                    for p in pts
                ]
                if not all(times[i] < times[i + 1]
                           for i in range(len(times) - 1)):
                    self._log.error(
                        "SAFETY: Rejecting trajectory — "
                        "timestamps not monotonic"
                    )
                    return False

            self._log.info(
                f"Executing trajectory on {robot_name} "
                f"({len(goal.trajectory.points)} points)"
                + (f" [retry {attempt}]" if attempt else "")
            )

            try:
                send_future = client.send_goal_async(goal)
                goal_handle = await await_ros_future(
                    send_future, timeout=10.0
                )
                if not goal_handle.accepted:
                    self._log.warn(
                        f"Trajectory goal rejected for {robot_name}"
                    )
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(5.0)
                        continue
                    return False

                self._log.info(
                    f"Trajectory accepted for {robot_name}, waiting…"
                )
                result_future = goal_handle.get_result_async()
                result = await await_ros_future(
                    result_future, timeout=timeout
                )
                code = result.result.error_code
                if code == FollowJointTrajectory.Result.SUCCESSFUL:
                    self._log.info(
                        f"Trajectory executed successfully on {robot_name}"
                    )
                    return True
                self._log.warn(
                    f"Trajectory error on {robot_name}: code={code}"
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(5.0)
                    continue
                return False

            except TimeoutError:
                self._log.error(
                    f"Trajectory timed out for {robot_name}"
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(5.0)
                    continue
                return False
            except Exception as e:
                self._log.error(
                    f"Trajectory error: {e}"
                )
                return False
        return False

    # ── Robot readiness check ────────────────────────────────────

    async def ensure_robot_ready(
        self, robot_name: str, timeout: float = 15.0,
    ) -> bool:
        """Ensure the robot is ready for trajectory execution."""
        mode = self._current_mode()
        if mode == "simulation":
            return True

        if RTDE_AVAILABLE and robot_name in self._rtde:
            rtde = self._rtde[robot_name]
            if rtde.is_connected:
                return True
            self._log.info(f"RTDE not connected — connecting {robot_name}…")
            return rtde.connect()

        # Fallback: check ROS2 robot program running state
        prog_running = getattr(self._node, '_robot_program_running', {})
        if prog_running.get(robot_name, False):
            return True

        self._log.warn(
            f"Robot program NOT running on {robot_name} — "
            f"attempting resend…"
        )
        resend_clients = getattr(
            self._node, '_resend_program_clients', {}
        )
        resend_client = resend_clients.get(robot_name)
        if resend_client and resend_client.service_is_ready():
            from std_srvs.srv import Trigger
            try:
                future = resend_client.call_async(Trigger.Request())
                await await_ros_future(future, timeout=5.0)
            except Exception as e:
                self._log.error(
                    f"resend_robot_program failed: {e}"
                )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if prog_running.get(robot_name, False):
                await asyncio.sleep(1.0)
                return True
            await asyncio.sleep(0.5)
        self._log.error(f"Timeout waiting for {robot_name}")
        return False
