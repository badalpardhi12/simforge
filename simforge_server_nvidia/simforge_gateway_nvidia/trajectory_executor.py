"""
Trajectory Executor — dispatches trajectories to RTDE or ROS2.

Handles:
  • RTDE servoJ streaming (real robot, 500 Hz)
  • ROS2 FollowJointTrajectory action (simulation)
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
        self._planner = None  # set via set_planner() after init

        # FollowJointTrajectory action clients (sim)
        self._traj_clients: Dict[str, ActionClient] = {}
        self._rebuild_traj_clients()

    # ── Planner injection ────────────────────────────────────────

    def set_planner(self, planner):
        """Wire in the cuRobo planner (called after both are created)."""
        self._planner = planner

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
        """Move a robot to its home position via cuRobo-planned trajectory.

        Uses cuRobo plan_to_joints for collision-aware path planning,
        then dispatches the trajectory via RTDE servoJ (real) or ROS2
        action (sim).  Falls back to raw RTDE moveJ only if cuRobo
        is unavailable.
        """
        cfg = ROBOT_CONFIG[robot_name]
        mode = self._current_mode()
        is_real = mode in ("real", "both")
        home = list(cfg["home_position"])

        # ── Try cuRobo collision-aware planning first ────────────
        if self._planner is not None and self._planner.has_robot(robot_name):
            # Get current joint positions
            current = None
            if is_real and RTDE_AVAILABLE and robot_name in self._rtde:
                rtde = self._rtde[robot_name]
                if not rtde.is_connected:
                    rtde.connect()
                if rtde.is_connected:
                    current = rtde.get_actual_q()

            if current is None and self._js_mgr is not None:
                st = self._js_mgr.robot_states.get(robot_name)
                if st and st.joint_positions and len(st.joint_positions) == 6:
                    current = list(st.joint_positions)

            if current is None:
                current = home  # already at home, nothing to do

            self._log.info(
                f"Planning collision-aware home path for {robot_name} "
                f"via cuRobo plan_to_joints"
            )

            result = await self._planner.plan_to_joints(
                robot_name,
                target_joints=home,
                current_joints=current,
                velocity_scaling=velocity_scaling,
            )

            if result is not None:
                ros_traj = self._planner.result_to_ros_trajectory(
                    result, robot_name,
                )
                if ros_traj is not None:
                    self._log.info(
                        f"Executing cuRobo-planned home trajectory "
                        f"for {robot_name}"
                    )
                    ok = await self.execute(robot_name, ros_traj)
                    if ok:
                        # Reconnect RTDE receive after trajectory
                        if is_real and RTDE_AVAILABLE and robot_name in self._rtde:
                            rtde = self._rtde[robot_name]
                            loop = asyncio.get_event_loop()
                            for attempt in range(3):
                                await loop.run_in_executor(
                                    None, rtde.reconnect_receive,
                                )
                                if rtde._recv_healthy:
                                    await asyncio.sleep(1.0)
                                    test_q = await loop.run_in_executor(
                                        None, rtde.get_actual_q,
                                    )
                                    if test_q is not None:
                                        break
                                    self._log.info(
                                        f"RTDE recv for {robot_name} "
                                        f"died again (attempt "
                                        f"{attempt + 1}/3), retrying..."
                                    )
                        return True
                    self._log.warn(
                        f"cuRobo-planned home trajectory execution "
                        f"failed for {robot_name}"
                    )
                    return False

            self._log.warn(
                f"cuRobo joint planning failed for home — "
                f"falling back to RTDE moveJ for {robot_name}"
            )

        # ── Fallback: raw RTDE moveJ (no collision avoidance) ────
        if is_real and RTDE_AVAILABLE and robot_name in self._rtde:
            rtde = self._rtde[robot_name]
            if not rtde.is_connected:
                rtde.connect()
            if rtde.is_connected:
                self._log.warn(
                    f"Moving {robot_name} home via RTDE moveJ "
                    f"(NO collision avoidance — cuRobo unavailable)"
                )
                loop = asyncio.get_event_loop()
                ok = await loop.run_in_executor(
                    None,
                    lambda: rtde.move_j(
                        home, speed=0.5, acceleration=0.5
                    ),
                )
                for attempt in range(3):
                    await loop.run_in_executor(
                        None, rtde.reconnect_receive,
                    )
                    if rtde._recv_healthy:
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

        # In real mode, position_callback publishes commanded positions
        # to /joint_states at ~50 Hz.  Do NOT fire the sim trajectory
        # in parallel — the active trajectory controller on fake
        # hardware would fight with the RTDE-sourced positions.

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

        # After successful servoJ, publish the robot's ACTUAL final
        # position to /joint_states so the sim model snaps to the
        # real-world position (eliminating visual jumps).
        if result and self._js_mgr is not None:
            actual_q = rtde.last_actual_q_post_exec
            if actual_q is not None:
                self._js_mgr.publish_positions(robot_name, actual_q)
                self._log.info(
                    f"Published actual post-exec position for "
                    f"{robot_name} to /joint_states"
                )

        return result

    # ── Actual position after execution ──────────────────────────

    def get_actual_position(self, robot_name: str):
        """Return the actual joint position after last RTDE execution.

        Returns None if RTDE not available or position not read.
        """
        if not RTDE_AVAILABLE:
            return None
        rtde = self._rtde.get(robot_name)
        if rtde is None:
            return None
        return rtde.last_actual_q_post_exec

    def get_actual_tcp_pose(self, robot_name: str):
        """Return actual TCP pose [x,y,z,rx,ry,rz] after last execution.

        Uses RTDE getActualTCPPose().  Position in metres, orientation
        as axis-angle rotation vector.  Returns None if unavailable.
        """
        if not RTDE_AVAILABLE:
            return None
        rtde = self._rtde.get(robot_name)
        if rtde is None:
            return None
        return rtde.get_actual_tcp_pose()

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
