#!/usr/bin/env python3
"""
Command Gateway Node for Valid8 Dual UR5e Cell

Bridges WebSocket commands from Mac client to ROS2.
Uses MoveIt2 service calls for:
  - IK solving  (/compute_ik)
  - Motion planning  (/plan_kinematic_path)
  - Trajectory execution  (FollowJointTrajectory action on the scaled_joint_trajectory_controller)

All planning and execution is server-side. The client only sends
Cartesian poses and the server handles IK → Plan → Execute.

Compatible with simforge_client RPC protocol.
"""

import asyncio
import json
import os
import time
import math
import traceback
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState

from control_msgs.action import FollowJointTrajectory

from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    RobotTrajectory,
)

import tf2_ros
from tf2_ros import Buffer, TransformListener

try:
    import websockets
    from websockets.server import serve
except ImportError:
    raise ImportError("websockets package required: pip install websockets>=12.0")

# ── Robot configuration ──────────────────────────────────────────────
ROBOT_CONFIG = {
    "nakul_ur5e": {
        "prefix": "nakul_",
        "ip": "192.168.1.9",
        "joints": [
            "nakul_shoulder_pan_joint",
            "nakul_shoulder_lift_joint",
            "nakul_elbow_joint",
            "nakul_wrist_1_joint",
            "nakul_wrist_2_joint",
            "nakul_wrist_3_joint",
        ],
        "controller": "nakul_scaled_joint_trajectory_controller",
        "planning_group": "nakul_arm",
        "ee_link": "nakul_tool0",
        # CRITICAL: The client generates poses for the tool tip (where the
        # camera/phone is mounted), NOT for tool0 (wrist flange).  IK must
        # target this link so the physical tool tip reaches the desired pose.
        "ik_tip_link": "nakul_tool_tip_link",
        "base_link": "nakul_base_link",
        "home_position": [0.0, -math.pi / 2, 0.0,
                          -math.pi / 2, 0.0, 0.0],
    },
    "sahadev_ur5e": {
        "prefix": "sahadev_",
        "ip": "192.168.1.16",
        "joints": [
            "sahadev_shoulder_pan_joint",
            "sahadev_shoulder_lift_joint",
            "sahadev_elbow_joint",
            "sahadev_wrist_1_joint",
            "sahadev_wrist_2_joint",
            "sahadev_wrist_3_joint",
        ],
        "controller": "sahadev_scaled_joint_trajectory_controller",
        "planning_group": "sahadev_arm",
        "ee_link": "sahadev_tool0",
        # sahadev has no tool mount — IK targets tool0 (wrist flange)
        "ik_tip_link": "sahadev_tool0",
        "base_link": "sahadev_base_link",
        "home_position": [0.0, -math.pi / 2, 0.0,
                          -math.pi / 2, 0.0, 0.0],
    },
}

KNOWN_OBJECTS = ["face_link", "table_link", "shop_floor"]

# ── Mode-switch signal files (shared with start_server.sh) ───
MODE_SWITCH_FILE = "/tmp/simforge_mode_switch"
CURRENT_MODE_FILE = "/tmp/simforge_current_mode"
STACK_READY_FILE = "/tmp/simforge_stack_ready"


# ── Helper dataclasses ───────────────────────────────────────────────


@dataclass
class ConnectedClient:
    client_id: str
    websocket: Any
    connected_at: float
    last_activity: float
    heartbeat_count: int = 0


@dataclass
class RobotStateInfo:
    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    last_update: float = 0.0


# ── Async ROS2 future helper ────────────────────────────────────────

async def await_ros_future(future, timeout: float = 10.0):
    """
    Poll an rclpy Future from an asyncio coroutine.

    The MultiThreadedExecutor is already spinning the node, so the
    future will be completed by the executor.  We just poll from
    asyncio to avoid blocking the event-loop or deadlocking with
    rclpy.spin_until_future_complete.
    """
    deadline = time.monotonic() + timeout
    while not future.done():
        if time.monotonic() > deadline:
            raise TimeoutError(f"ROS future timed out after {timeout}s")
        await asyncio.sleep(0.05)
    return future.result()


# ── Main node ────────────────────────────────────────────────────────


class CommandGatewayNode(Node):
    """WebSocket ↔ ROS2 bridge with MoveIt IK + planning."""

    def __init__(self):
        super().__init__("command_gateway")

        self.declare_parameter("websocket_port", 8766)
        self.declare_parameter("websocket_host", "0.0.0.0")
        self.declare_parameter("max_clients", 5)

        self.ws_port = self.get_parameter("websocket_port").value
        self.ws_host = self.get_parameter("websocket_host").value
        self.max_clients = self.get_parameter("max_clients").value

        self.cb_group = ReentrantCallbackGroup()

        # Connected websocket clients
        # Named _ws_clients to avoid collision with rclpy's internal _clients
        self._ws_clients: Dict[str, ConnectedClient] = {}

        # Robot joint-state cache
        self._robot_states: Dict[str, RobotStateInfo] = {
            n: RobotStateInfo() for n in ROBOT_CONFIG
        }

        # Proto-sim flags
        self._proto_sim_running = False
        self._proto_sim_stop = False

        # Hardware mode tracking
        self._current_mode = self._read_current_mode()

        # ── Publishers ───────────────────────────────────────────────
        self.heartbeat_pub = self.create_publisher(String, "/safety/heartbeat", 10)
        self.estop_pub = self.create_publisher(String, "/safety/emergency_stop", 10)

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 10
        )

        # ── Action clients (FollowJointTrajectory) ───────────────────
        self._traj_clients: Dict[str, ActionClient] = {}
        for robot_name, cfg in ROBOT_CONFIG.items():
            action_ns = f"/{cfg['controller']}/follow_joint_trajectory"
            self._traj_clients[robot_name] = ActionClient(
                self, FollowJointTrajectory, action_ns,
                callback_group=self.cb_group,
            )
            self.get_logger().info(
                f"Trajectory action client for {robot_name}: {action_ns}"
            )

        # ── MoveIt service clients ───────────────────────────────────
        self._ik_client = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=self.cb_group
        )
        self._plan_client = self.create_client(
            GetMotionPlan, "/plan_kinematic_path", callback_group=self.cb_group
        )

        # ── Robot-program-running state (real hardware only) ─────────
        # The UR driver's io_and_status_controller publishes whether the
        # URScript program is currently running on the robot.  Trajectory
        # commands are silently dropped by the hardware interface when
        # the program is NOT running.
        # In headless mode, the program can be re-sent via the
        # resend_robot_program service.
        self._robot_program_running: Dict[str, bool] = {
            n: False for n in ROBOT_CONFIG
        }
        self._resend_program_clients: Dict[str, Any] = {}
        for robot_name, cfg in ROBOT_CONFIG.items():
            prefix = cfg["prefix"]
            # Subscribe to program running state
            topic = f"/{prefix}io_and_status_controller/robot_program_running"
            self.create_subscription(
                Bool, topic,
                lambda msg, rn=robot_name: self._on_robot_program_running(rn, msg),
                10,
            )
            self.get_logger().info(
                f"Subscribed to robot_program_running for {robot_name}: {topic}"
            )
            # Service client for resending the URScript program
            srv_name = f"/{prefix}io_and_status_controller/resend_robot_program"
            self._resend_program_clients[robot_name] = self.create_client(
                Trigger, srv_name, callback_group=self.cb_group
            )
            self.get_logger().info(
                f"Resend program service for {robot_name}: {srv_name}"
            )

        # ── TF2 ──────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ws_server = None

        self.get_logger().info(
            f"Command Gateway initialised – WS on {self.ws_host}:{self.ws_port}"
        )
        self.get_logger().info(f"Robots: {list(ROBOT_CONFIG.keys())}")

    # ─────────────────────────────────────────────────────────────────
    # Joint state callback
    # ─────────────────────────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState):
        for robot_name, cfg in ROBOT_CONFIG.items():
            positions, velocities = [], []
            for jn in cfg["joints"]:
                if jn in msg.name:
                    idx = msg.name.index(jn)
                    positions.append(msg.position[idx])
                    velocities.append(
                        msg.velocity[idx] if idx < len(msg.velocity) else 0.0
                    )
            if len(positions) == 6:
                st = self._robot_states[robot_name]
                st.joint_positions = positions
                st.joint_velocities = velocities
                st.last_update = time.time()

    # ─────────────────────────────────────────────────────────────────
    # Robot-program-running state (real hardware)
    # ─────────────────────────────────────────────────────────────────

    def _on_robot_program_running(self, robot_name: str, msg: Bool):
        """Callback for {prefix}_io_and_status_controller/robot_program_running."""
        prev = self._robot_program_running.get(robot_name, False)
        self._robot_program_running[robot_name] = msg.data
        if msg.data != prev:
            self.get_logger().info(
                f"Robot program running [{robot_name}]: {msg.data}"
            )

    async def _ensure_robot_ready(
        self, robot_name: str, timeout: float = 15.0
    ) -> bool:
        """
        Ensure the UR robot program is running before executing a trajectory.

        In headless mode, the UR driver sends a URScript program to the robot
        when the hardware interface activates.  If the program stops (e.g.
        protective stop, teach pendant interaction), the io_and_status_controller
        publishes False on robot_program_running.  The hardware interface's
        write() will then refuse to send motion commands, causing trajectory
        timeouts.

        This method checks the flag, calls resend_robot_program if needed,
        and waits until the flag becomes True.

        In simulation mode, the flag is always False (no io_and_status_controller)
        but robot_program_running isn't needed — so we skip the check.
        """
        current_mode = self._read_current_mode()
        if current_mode == "simulation":
            # In simulation, robot_program_running doesn't exist / isn't needed
            return True

        # Already running?
        if self._robot_program_running.get(robot_name, False):
            return True

        self.get_logger().warn(
            f"Robot program NOT running on {robot_name} — "
            f"attempting resend_robot_program…"
        )

        # Try to call resend_robot_program
        resend_client = self._resend_program_clients.get(robot_name)
        if resend_client and resend_client.service_is_ready():
            try:
                future = resend_client.call_async(Trigger.Request())
                result = await await_ros_future(future, timeout=5.0)
                if result.success:
                    self.get_logger().info(
                        f"resend_robot_program succeeded for {robot_name}"
                    )
                else:
                    self.get_logger().warn(
                        f"resend_robot_program returned non-success "
                        f"for {robot_name}: {result.message}"
                    )
            except Exception as e:
                self.get_logger().error(
                    f"resend_robot_program failed for {robot_name}: {e}"
                )
        else:
            self.get_logger().warn(
                f"resend_robot_program service not available for {robot_name}"
            )

        # Wait for robot_program_running to become True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._robot_program_running.get(robot_name, False):
                self.get_logger().info(
                    f"Robot program now running on {robot_name} "
                    f"— ready for trajectory execution"
                )
                # Extra settle time for controller_stopper to reactivate controllers
                await asyncio.sleep(1.0)
                return True
            await asyncio.sleep(0.5)

        self.get_logger().error(
            f"Timeout ({timeout}s) waiting for robot_program_running "
            f"on {robot_name}"
        )
        return False

    # ─────────────────────────────────────────────────────────────────
    # MoveIt helpers (IK, plan, execute)
    # ─────────────────────────────────────────────────────────────────

    async def _wait_for_moveit(self, timeout: float = 30.0) -> bool:
        """Wait until /compute_ik and /plan_kinematic_path are reachable."""
        self.get_logger().info("Waiting for MoveIt services…")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ik_ok = self._ik_client.service_is_ready()
            plan_ok = self._plan_client.service_is_ready()
            if ik_ok and plan_ok:
                self.get_logger().info("MoveIt services are ready")
                return True
            await asyncio.sleep(0.5)
        self.get_logger().error("MoveIt services not available within timeout")
        return False

    async def _solve_ik(
        self, robot_name: str, pose: Pose, seed_joints: Optional[List[float]] = None
    ) -> Optional[List[float]]:
        """Call /compute_ik and return joint positions or None."""
        cfg = ROBOT_CONFIG[robot_name]

        req = GetPositionIK.Request()
        req.ik_request.group_name = cfg["planning_group"]
        req.ik_request.avoid_collisions = True

        # Pose target — use the robot's base_link as frame, matching
        # how the old server works.  The client generates poses in the
        # reference frame returned by get_environment_info, which is
        # now the robot's base_link.
        ps = PoseStamped()
        ps.header.frame_id = cfg["base_link"]
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = pose
        req.ik_request.pose_stamped = ps

        # Use ik_tip_link (tool_tip_link) — NOT ee_link (tool0).
        # The client generates poses for the actual tool tip where the
        # camera / iPhone is mounted.  Solving IK for tool0 would place
        # the wrist flange at the target instead of the tool tip, causing
        # an offset equal to the tool mount geometry.
        req.ik_request.ik_link_name = cfg["ik_tip_link"]

        # Seed state
        seed = seed_joints or self._robot_states[robot_name].joint_positions
        if not seed or len(seed) != 6:
            seed = list(cfg["home_position"])
        req.ik_request.robot_state.joint_state.name = list(cfg["joints"])
        req.ik_request.robot_state.joint_state.position = list(seed)

        req.ik_request.timeout.sec = 2
        req.ik_request.timeout.nanosec = 0

        try:
            future = self._ik_client.call_async(req)
            result = await await_ros_future(future, timeout=10.0)
            if result.error_code.val == 1:  # MoveItErrorCodes.SUCCESS
                joint_map = dict(
                    zip(
                        result.solution.joint_state.name,
                        result.solution.joint_state.position,
                    )
                )
                joints = [joint_map[j] for j in cfg["joints"]]
                # Normalize to the 2π-equivalent closest to the seed so
                # the planner doesn't sweep through unnecessary arcs.
                joints = self._normalize_joint_angles(joints, seed)
                return joints
            else:
                self.get_logger().warn(
                    f"IK failed for {robot_name}: error_code={result.error_code.val}"
                )
                return None
        except TimeoutError:
            self.get_logger().error(f"IK service call timed out for {robot_name}")
            return None
        except Exception as exc:
            self.get_logger().error(f"IK service call failed: {exc}")
            return None

    @staticmethod
    def _normalize_joint_angles(
        joints: List[float],
        seed: Optional[List[float]] = None,
    ) -> List[float]:
        """Normalize joint angles to the 2π-equivalent closest to the seed.

        UR joints are continuous and IK may return values like 4.53 rad
        which is equivalent to 4.53 - 2π ≈ -1.75 rad.  If the seed
        (current robot state) is -1.57 rad, the -1.75 solution is much
        closer and avoids a huge sweep through collision space.
        """
        TWO_PI = 2.0 * math.pi
        result = list(joints)
        if seed is None:
            for i in range(len(result)):
                while result[i] > math.pi:
                    result[i] -= TWO_PI
                while result[i] < -math.pi:
                    result[i] += TWO_PI
            return result
        for i in range(min(len(result), len(seed))):
            diff = result[i] - seed[i]
            k = round(diff / TWO_PI)
            result[i] -= k * TWO_PI
        return result

    async def _plan_to_joints(
        self,
        robot_name: str,
        target_joints: List[float],
        velocity_scaling: float = 0.3,
        acceleration_scaling: float = 0.3,
    ) -> Optional[RobotTrajectory]:
        """Call /plan_kinematic_path and return RobotTrajectory or None."""
        cfg = ROBOT_CONFIG[robot_name]

        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = cfg["planning_group"]
        mp.num_planning_attempts = 20
        mp.allowed_planning_time = 10.0
        mp.max_velocity_scaling_factor = velocity_scaling
        mp.max_acceleration_scaling_factor = acceleration_scaling

        # Workspace bounds (matching old server: ±2m XY, -0.5 to 3m Z)
        mp.workspace_parameters.header.frame_id = cfg["base_link"]
        mp.workspace_parameters.min_corner.x = -2.0
        mp.workspace_parameters.min_corner.y = -2.0
        mp.workspace_parameters.min_corner.z = -0.5
        mp.workspace_parameters.max_corner.x = 2.0
        mp.workspace_parameters.max_corner.y = 2.0
        mp.workspace_parameters.max_corner.z = 3.0

        # Start state = current joints
        current = self._robot_states[robot_name].joint_positions
        if current and len(current) == 6:
            mp.start_state.joint_state.name = list(cfg["joints"])
            mp.start_state.joint_state.position = list(current)
            mp.start_state.is_diff = False

        # Goal constraints = target joints
        constraints = Constraints()
        for jn, val in zip(cfg["joints"], target_joints):
            jc = JointConstraint()
            jc.joint_name = jn
            jc.position = val
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        mp.goal_constraints.append(constraints)

        try:
            future = self._plan_client.call_async(req)
            result = await await_ros_future(future, timeout=15.0)
            if result.motion_plan_response.error_code.val == 1:  # SUCCESS
                pts = result.motion_plan_response.trajectory.joint_trajectory.points
                self.get_logger().info(
                    f"Motion plan for {robot_name}: {len(pts)} waypoints"
                )
                return result.motion_plan_response.trajectory
            else:
                self.get_logger().warn(
                    f"Planning failed for {robot_name}: "
                    f"error_code={result.motion_plan_response.error_code.val}"
                )
                return None
        except TimeoutError:
            self.get_logger().error(f"Planning service timed out for {robot_name}")
            return None
        except Exception as exc:
            self.get_logger().error(f"Planning service call failed: {exc}")
            return None

    async def _execute_trajectory(
        self, robot_name: str, trajectory: RobotTrajectory, timeout: float = 60.0
    ) -> bool:
        """Send a RobotTrajectory to the FollowJointTrajectory action."""
        client = self._traj_clients.get(robot_name)
        if client is None:
            self.get_logger().error(f"No trajectory client for {robot_name}")
            return False

        # Wait for action server (non-blocking poll)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if client.server_is_ready():
                break
            await asyncio.sleep(0.2)
        else:
            self.get_logger().error(
                f"Trajectory action server not available for {robot_name}"
            )
            return False

        # ── Ensure the UR robot program is running ──────────────
        # Without this, the hardware interface silently drops all
        # motion commands and the trajectory action times out.
        ready = await self._ensure_robot_ready(robot_name, timeout=15.0)
        if not ready:
            self.get_logger().error(
                f"Robot program not running on {robot_name} — "
                f"cannot execute trajectory"
            )
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory.joint_trajectory

        # ── Fix: strip duplicate-time leading point ─────────────
        # MoveIt often produces trajectories whose first point has
        # time_from_start == 0 (the current state).  The UR
        # scaled_joint_trajectory_controller requires *strictly
        # increasing* timestamps, so we drop the first point if it
        # shares a timestamp with the second.
        pts = goal.trajectory.points
        if len(pts) >= 2:
            t0 = pts[0].time_from_start.sec + pts[0].time_from_start.nanosec * 1e-9
            t1 = pts[1].time_from_start.sec + pts[1].time_from_start.nanosec * 1e-9
            if t0 >= t1 or t0 == 0.0:
                goal.trajectory.points = list(pts[1:])

        n_pts = len(goal.trajectory.points)
        self.get_logger().info(f"Executing trajectory on {robot_name} ({n_pts} points)")

        try:
            # Send goal
            send_future = client.send_goal_async(goal)
            goal_handle = await await_ros_future(send_future, timeout=10.0)

            if not goal_handle.accepted:
                self.get_logger().warn(f"Trajectory goal rejected for {robot_name}")
                return False

            self.get_logger().info(f"Trajectory accepted for {robot_name}, waiting…")

            # Wait for result
            result_future = goal_handle.get_result_async()
            result = await await_ros_future(result_future, timeout=timeout)

            error_code = result.result.error_code
            if error_code == FollowJointTrajectory.Result.SUCCESSFUL:
                self.get_logger().info(
                    f"Trajectory executed successfully on {robot_name}"
                )
                return True
            else:
                self.get_logger().warn(
                    f"Trajectory execution error on {robot_name}: code={error_code}"
                )
                return False
        except TimeoutError:
            self.get_logger().error(
                f"Trajectory execution timed out for {robot_name}"
            )
            return False
        except Exception as exc:
            self.get_logger().error(
                f"Trajectory execution failed for {robot_name}: {exc}\n"
                f"{traceback.format_exc()}"
            )
            return False

    async def _move_to_pose(
        self,
        robot_name: str,
        position: List[float],
        orientation: List[float],
        velocity_scaling: float = 0.3,
    ) -> str:
        """
        Full pipeline: IK → Plan → Execute.
        Tries up to 3 IK seeds if the first solution can't be reached by OMPL.
        Returns: "success", "ik_failed", "plan_failed", or "exec_failed".
        """
        import random

        pose = Pose()
        pose.position = Point(x=position[0], y=position[1], z=position[2])
        pose.orientation = Quaternion(
            x=orientation[0], y=orientation[1],
            z=orientation[2], w=orientation[3],
        )

        cfg = ROBOT_CONFIG[robot_name]
        max_ik_attempts = 3

        # Build a list of IK seeds: current state, home, random perturbations
        current = self._robot_states[robot_name].joint_positions
        seeds: List[Optional[List[float]]] = [
            current if current and len(current) == 6 else None,
            list(cfg["home_position"]),
            None,  # Let the IK solver pick a random seed
        ]

        for attempt_idx, seed in enumerate(seeds[:max_ik_attempts]):
            # 1. Solve IK with this seed
            joint_goal = await self._solve_ik(robot_name, pose, seed_joints=seed)
            if joint_goal is None:
                self.get_logger().info(
                    f"IK attempt {attempt_idx+1}/{max_ik_attempts} failed for {robot_name}"
                )
                continue

            self.get_logger().info(
                f"IK solution (attempt {attempt_idx+1}) for {robot_name}: "
                f"{[f'{v:.3f}' for v in joint_goal]}"
            )

            # 2. Plan trajectory
            trajectory = await self._plan_to_joints(
                robot_name, joint_goal, velocity_scaling=velocity_scaling
            )
            if trajectory is None:
                self.get_logger().info(
                    f"Planning attempt {attempt_idx+1}/{max_ik_attempts} failed for {robot_name}, trying different IK seed"
                )
                continue

            # 3. Execute trajectory
            ok = await self._execute_trajectory(robot_name, trajectory)
            return "success" if ok else "exec_failed"

        # All attempts exhausted
        return "plan_failed"

    async def _move_to_home(self, robot_name: str) -> bool:
        """Plan and execute a return-to-home motion."""
        cfg = ROBOT_CONFIG[robot_name]
        trajectory = await self._plan_to_joints(
            robot_name, cfg["home_position"],
            velocity_scaling=0.3, acceleration_scaling=0.3,
        )
        if trajectory is None:
            self.get_logger().warn(f"Cannot plan home for {robot_name}")
            return False
        return await self._execute_trajectory(robot_name, trajectory)

    # ─────────────────────────────────────────────────────────────────
    # WebSocket server
    # ─────────────────────────────────────────────────────────────────

    async def start_websocket_server(self):
        self.get_logger().info(
            f"Starting WebSocket server on {self.ws_host}:{self.ws_port}"
        )
        self.ws_server = await serve(
            self._handle_client, self.ws_host, self.ws_port,
            ping_interval=30, ping_timeout=300,
        )
        self.get_logger().info("WebSocket server started")

    async def _handle_client(self, websocket, path: str = None):
        cid = f"client_{id(websocket)}"
        if len(self._ws_clients) >= self.max_clients:
            self.get_logger().warning(f"Max clients – rejecting {cid}")
            await websocket.close(1013, "Max clients reached")
            return

        client = ConnectedClient(
            client_id=cid, websocket=websocket,
            connected_at=time.time(), last_activity=time.time(),
        )
        self._ws_clients[cid] = client
        self.get_logger().info(f"Client connected: {cid}")

        try:
            async for message in websocket:
                await self._process_message(client, message)
        except websockets.ConnectionClosed as e:
            self.get_logger().info(f"Client disconnected: {cid} – {e}")
        except Exception as e:
            self.get_logger().error(f"Client error: {cid} – {e}")
        finally:
            self._ws_clients.pop(cid, None)
            self.get_logger().info(f"Client removed: {cid}")

    async def _process_message(self, client: ConnectedClient, raw: str):
        try:
            msg = json.loads(raw)
            client.last_activity = time.time()
            t = msg.get("type", "")

            if t == "heartbeat":
                await self._on_heartbeat(client, msg)
            elif t == "ping":
                await self._on_ping(client, msg)
            elif t == "rpc":
                await self._on_rpc(client, msg)
            elif t == "move_robot":
                await self._on_move_robot(client, msg)
            elif t == "emergency_stop":
                await self._on_estop(client, msg)
            elif t == "soft_stop":
                self._proto_sim_stop = True
                await self._reply(client, msg.get("request_id"),
                                  success=True, message="Soft stop requested")
            elif t == "get_robot_state":
                await self._on_get_state(client, msg)
            else:
                await self._send_error(client, msg.get("request_id"),
                                       f"Unknown message type: {t}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Bad JSON from {client.client_id}: {e}")
        except Exception as e:
            self.get_logger().error(
                f"Error processing msg: {e}\n{traceback.format_exc()}"
            )
            await self._send_error(client, None, str(e))

    # ─────────────────────────────────────────────────────────────────
    # Simple message handlers
    # ─────────────────────────────────────────────────────────────────

    async def _on_heartbeat(self, client, msg):
        client.heartbeat_count += 1
        hb = String()
        hb.data = (
            f"{client.client_id}:{msg.get('sequence', 0)}:"
            f"{msg.get('latency_ms', 0)}"
        )
        self.heartbeat_pub.publish(hb)
        await client.websocket.send(json.dumps({
            "type": "heartbeat_ack",
            "request_id": msg.get("request_id"),
            "sequence": msg.get("sequence", 0),
            "server_time_ns": time.time_ns(),
        }))

    async def _on_ping(self, client, msg):
        await client.websocket.send(json.dumps({
            "type": "pong",
            "request_id": msg.get("request_id"),
            "timestamp_ns": time.time_ns(),
        }))

    async def _on_estop(self, client, msg):
        rname = msg.get("robot")
        self.get_logger().warning(f"EMERGENCY STOP: {rname or 'ALL'}")
        m = String()
        m.data = rname or "all"
        self.estop_pub.publish(m)
        self._proto_sim_stop = True
        await client.websocket.send(json.dumps({
            "type": "emergency_stop_active",
            "request_id": msg.get("request_id"),
            "robots_stopped": (
                [rname] if rname else list(ROBOT_CONFIG.keys())
            ),
        }))

    async def _on_get_state(self, client, msg):
        rn = msg.get("robot", list(ROBOT_CONFIG.keys())[0])
        cfg = ROBOT_CONFIG.get(rn)
        if not cfg:
            await self._send_error(
                client, msg.get("request_id"), f"Unknown robot: {rn}"
            )
            return
        st = self._robot_states[rn]
        await client.websocket.send(json.dumps({
            "type": "robot_state",
            "request_id": msg.get("request_id"),
            "robot": rn,
            "joint_names": cfg["joints"],
            "joint_positions": st.joint_positions,
            "joint_velocities": st.joint_velocities,
            "last_update": st.last_update,
        }))

    async def _on_move_robot(self, client, msg):
        rid = msg.get("request_id")
        rn = msg.get("robot_name", list(ROBOT_CONFIG.keys())[0])
        target = msg.get("target_joints")
        if rn not in ROBOT_CONFIG:
            await self._send_error(client, rid, f"Unknown robot: {rn}")
            return
        if not target or len(target) != 6:
            await self._send_error(
                client, rid, "target_joints with 6 values required"
            )
            return
        traj = await self._plan_to_joints(rn, target)
        if traj is None:
            await self._send_error(client, rid, "Planning failed")
            return
        ok = await self._execute_trajectory(rn, traj)
        await self._reply(client, rid, success=ok)

    # ─────────────────────────────────────────────────────────────────
    # RPC dispatch
    # ─────────────────────────────────────────────────────────────────

    async def _on_rpc(self, client, msg):
        rid = msg.get("request_id")
        method = msg.get("method", "")
        params = msg.get("params", {})
        self.get_logger().info(f"RPC from {client.client_id}: {method}")

        try:
            if method == "get_environment_info":
                result = await self._rpc_env_info(params)
            elif method == "get_robot_status":
                result = await self._rpc_robot_status(params)
            elif method == "prepare_mode":
                result = await self._rpc_prepare_mode(params)
            elif method == "run_proto_sim":
                # Long-running — fire and forget, sends its own result
                asyncio.ensure_future(
                    self._rpc_run_proto_sim(client, rid, params)
                )
                return
            elif method == "stop_proto_sim":
                self._proto_sim_stop = True
                result = {"success": True, "message": "Stop requested"}
            elif method == "check_collision":
                result = {"success": True, "in_collision": False}
            elif method == "plan_motion":
                result = await self._rpc_plan_motion(params)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown RPC method: {method}",
                }

            await client.websocket.send(json.dumps({
                "type": "rpc_result", "request_id": rid, **result,
            }))
        except Exception as e:
            self.get_logger().error(
                f"RPC error ({method}): {e}\n{traceback.format_exc()}"
            )
            await self._send_error(client, rid, str(e))

    # ─────────────────────────────────────────────────────────────────
    # RPC implementations
    # ─────────────────────────────────────────────────────────────────

    async def _rpc_env_info(self, params):
        # Use the nakul robot's base_link as the reference frame,
        # matching how the old server works.  The client generates poses
        # in this frame, and IK/planning also operates in this frame.
        default_robot = list(ROBOT_CONFIG.keys())[0]
        ref = ROBOT_CONFIG[default_robot]["base_link"]
        obj_tf = {}
        for obj in KNOWN_OBJECTS:
            try:
                t = self.tf_buffer.lookup_transform(
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
                self.get_logger().warn(f"TF for {obj}: {e}")
                obj_tf[obj] = None
        return {
            "success": True,
            "robots": list(ROBOT_CONFIG.keys()),
            "objects": KNOWN_OBJECTS,
            "object_transforms": obj_tf,
            "reference_frame": ref,
        }

    async def _rpc_robot_status(self, params):
        rn = params.get("robot_name") or list(ROBOT_CONFIG.keys())[0]
        cfg = ROBOT_CONFIG.get(rn)
        if not cfg:
            return {"success": False, "error": f"Unknown robot: {rn}"}
        st = self._robot_states.get(rn, RobotStateInfo())
        has_joints = len(st.joint_positions) == 6

        traj_ok = rn in self._traj_clients and self._traj_clients[rn].server_is_ready()
        moveit_ok = (
            self._ik_client.service_is_ready()
            and self._plan_client.service_is_ready()
        )

        current_mode = self._read_current_mode()
        is_real = current_mode in ("real", "both")
        prog_running = self._robot_program_running.get(rn, False)

        return {
            "success": True,
            "real_robot_available": is_real and traj_ok and has_joints,
            "simulation_available": True,
            "available_modes": ["simulation", "real"],
            "current_mode": current_mode,
            "connection_details": {
                "follow_trajectory_action": (
                    "available" if traj_ok else "not_available"
                ),
                "moveit": "available" if moveit_ok else "not_available",
                "robot_program_running": prog_running,
            },
            "current_joint_positions": (
                st.joint_positions if has_joints else list(cfg["home_position"])
            ),
            "position_source": "real_robot" if (is_real and has_joints) else "default",
        }

    # ── Mode switching helpers ───────────────────────────────────────

    @staticmethod
    def _read_current_mode() -> str:
        """Read the current hardware mode from the signal file."""
        try:
            if os.path.exists(CURRENT_MODE_FILE):
                with open(CURRENT_MODE_FILE, "r") as f:
                    return f.read().strip()
        except Exception:
            pass
        return "simulation"

    def _request_mode_switch(self, mode: str) -> None:
        """Write a mode-switch request for start_server.sh to pick up."""
        with open(MODE_SWITCH_FILE, "w") as f:
            f.write(mode)

    async def _wait_for_ros_stack(
        self,
        target_mode: str,
        timeout: float = 120.0,
        progress_cb=None,
    ) -> dict:
        """Wait for the ROS2 stack to come back up after a mode switch.

        Uses a robust multi-phase approach:
          Phase 1 – Wait for supervisor to consume the mode-switch signal file
                   AND clear the old readiness signals (STACK_READY_FILE gone,
                   CURRENT_MODE_FILE cleared).
          Phase 2 – Wait for STACK_READY_FILE to appear (written by
                   start_server.sh only after joint_states, robot_description,
                   and move_group are publishing).
          Phase 3 – Verify CURRENT_MODE_FILE matches the target mode.
          Phase 4 – Wait for fresh joint states on the gateway side
                   (timestamp > switch_start_time).
          Phase 5 – Verify MoveIt services are responsive.
          Phase 6 – (Real mode only) Wait for robot_program_running.

        Returns a dict with {"ok": bool, "phase": str, "message": str}.
        """
        switch_start = time.monotonic()
        overall_deadline = switch_start + timeout

        async def _report(phase: str, detail: str):
            msg = f"[{phase}] {detail}"
            self.get_logger().info(msg)
            if progress_cb:
                try:
                    await progress_cb(phase, detail)
                except Exception:
                    pass

        # ── Phase 1: wait for supervisor to pick up the signal ───────
        await _report("Phase 1/6", "Waiting for supervisor to pick up mode-switch signal…")
        phase1_deadline = min(switch_start + 30.0, overall_deadline)
        while time.monotonic() < phase1_deadline:
            signal_gone = not os.path.exists(MODE_SWITCH_FILE)
            ready_gone = not os.path.exists(STACK_READY_FILE)
            if signal_gone and ready_gone:
                await _report("Phase 1/6",
                    "Supervisor picked up signal — old stack tearing down")
                break
            await asyncio.sleep(0.5)
        else:
            # Even if we timed out here, continue — maybe the signal
            # was already consumed and we missed it.
            await _report("Phase 1/6",
                "Signal wait timed out — proceeding to Phase 2")

        # ── Phase 2: wait for STACK_READY_FILE ──────────────────────
        await _report("Phase 2/6", "Waiting for new ROS2 stack to become healthy…")
        phase2_deadline = min(time.monotonic() + 90.0, overall_deadline)
        stack_ready = False
        while time.monotonic() < phase2_deadline:
            if os.path.exists(STACK_READY_FILE):
                await _report("Phase 2/6",
                    "Stack ready signal received — supervisor health check passed")
                stack_ready = True
                break
            await asyncio.sleep(1.0)
        if not stack_ready:
            return {
                "ok": False,
                "phase": "Phase 2/6",
                "message": (
                    "Timed out waiting for ROS2 stack to become healthy. "
                    "The supervisor health check (joint_states, robot_description, "
                    "move_group) did not pass within 90s."),
            }

        # ── Phase 3: verify CURRENT_MODE_FILE matches target ────────
        await _report("Phase 3/6", "Verifying mode file matches target…")
        file_mode = self._read_current_mode()
        target_is_real = target_mode in ("real", "both")
        file_is_real = file_mode in ("real", "both")
        if target_is_real != file_is_real:
            return {
                "ok": False,
                "phase": "Phase 3/6",
                "message": (
                    f"Mode mismatch: requested '{target_mode}' but "
                    f"supervisor reported '{file_mode}'"),
            }
        await _report("Phase 3/6", f"Mode file confirmed: {file_mode}")

        # ── Phase 4: wait for fresh joint states ────────────────────
        await _report("Phase 4/6", "Waiting for fresh joint states…")
        phase4_deadline = min(time.monotonic() + 20.0, overall_deadline)
        while time.monotonic() < phase4_deadline:
            all_fresh = True
            for rn in ROBOT_CONFIG:
                st = self._robot_states.get(rn)
                if not st or st.last_update < switch_start or len(st.joint_positions) != 6:
                    all_fresh = False
                    break
            if all_fresh:
                await _report("Phase 4/6",
                    "Fresh joint states received for all robots")
                break
            await asyncio.sleep(0.5)
        else:
            # Log which robots are missing
            missing = []
            for rn in ROBOT_CONFIG:
                st = self._robot_states.get(rn)
                if not st or st.last_update < switch_start or len(st.joint_positions) != 6:
                    missing.append(rn)
            await _report("Phase 4/6",
                f"Joint state timeout — missing: {missing}. Proceeding anyway.")

        # ── Phase 5: verify MoveIt services respond ─────────────────
        await _report("Phase 5/6", "Verifying MoveIt services…")
        phase5_deadline = min(time.monotonic() + 15.0, overall_deadline)
        while time.monotonic() < phase5_deadline:
            ik_ok = self._ik_client.service_is_ready()
            plan_ok = self._plan_client.service_is_ready()
            all_traj_ok = all(
                tc.server_is_ready() for tc in self._traj_clients.values()
            )
            if ik_ok and plan_ok and all_traj_ok:
                await _report("Phase 5/6",
                    "All MoveIt services and trajectory servers ready")
                break
            await asyncio.sleep(1.0)
        else:
            not_ready = []
            if not self._ik_client.service_is_ready():
                not_ready.append("compute_ik")
            if not self._plan_client.service_is_ready():
                not_ready.append("plan_kinematic_path")
            for rname, tc in self._traj_clients.items():
                if not tc.server_is_ready():
                    not_ready.append(f"{rname}_trajectory")
            return {
                "ok": False,
                "phase": "Phase 5/6",
                "message": (
                    f"MoveIt services not ready after stack health passed. "
                    f"Missing: {not_ready}"),
            }

        # ── Phase 6: (real mode) wait for robot programs ────────────
        current_mode = self._read_current_mode()
        if current_mode in ("real", "both"):
            await _report("Phase 6/6",
                "Waiting for robot programs to start running…")
            phase6_deadline = min(time.monotonic() + 30.0, overall_deadline)
            while time.monotonic() < phase6_deadline:
                all_running = all(
                    self._robot_program_running.get(rn, False)
                    for rn in ROBOT_CONFIG
                )
                if all_running:
                    await _report("Phase 6/6",
                        "All robot programs running")
                    break
                await asyncio.sleep(0.5)
            else:
                for rn in ROBOT_CONFIG:
                    if not self._robot_program_running.get(rn, False):
                        self.get_logger().warn(
                            f"Robot program NOT running on {rn} — "
                            f"will retry via resend_robot_program before "
                            f"first trajectory"
                        )
        else:
            await _report("Phase 6/6", "Simulation mode — skipping robot program check")

        elapsed = time.monotonic() - switch_start
        await _report("Complete",
            f"Stack ready in {elapsed:.1f}s")
        return {
            "ok": True,
            "phase": "Complete",
            "message": f"ROS2 stack ready in {elapsed:.1f}s",
        }

    async def _rpc_prepare_mode(self, params):
        """Handle mode switching between simulation and real robot.

        When the requested mode differs from the current mode, this
        writes a signal file that the supervisor script (start_server.sh)
        picks up.  The supervisor tears down the current ROS2 stack
        (ros2_control_node, move_group, robot_state_publisher, controllers)
        and restarts it with the appropriate ``use_fake_hardware`` setting.

        The gateway node itself stays alive throughout because it runs as
        a separate process managed by the supervisor.
        """
        mode = params.get("mode", "simulation")
        self.get_logger().info(f"Preparing mode: {mode}")

        current_mode = self._read_current_mode()
        # Normalize: "both" is treated like "real" for hardware purposes
        need_real = mode in ("real", "both")
        have_real = current_mode in ("real", "both")

        if need_real != have_real:
            # ── Mode switch required ─────────────────────────────────
            if self._proto_sim_running:
                return {
                    "success": False,
                    "ready": False,
                    "message": "Cannot switch mode while a protocol is running",
                    "can_retry": True,
                }

            self.get_logger().info(
                f"Mode switch: {current_mode} → {mode}  "
                f"(use_fake_hardware: {not need_real})"
            )

            # Invalidate cached joint state timestamps so Phase 4 of
            # _wait_for_ros_stack will wait for genuinely new data from
            # the new stack.
            for rn in ROBOT_CONFIG:
                st = self._robot_states.get(rn)
                if st:
                    st.last_update = 0.0

            # Reset robot_program_running flags (old stack is going away)
            for rn in ROBOT_CONFIG:
                self._robot_program_running[rn] = False

            # Signal the supervisor script
            self._request_mode_switch(mode)

            # Collect progress phases for the client
            progress_log: list = []

            async def _progress_cb(phase: str, detail: str):
                progress_log.append({"phase": phase, "detail": detail})

            # Wait for the new stack to come up with full verification
            result = await self._wait_for_ros_stack(
                target_mode=mode, timeout=120.0, progress_cb=_progress_cb,
            )

            if not result["ok"]:
                return {
                    "success": True,
                    "ready": False,
                    "message": (
                        f"Mode switch to {mode} failed at {result['phase']}: "
                        f"{result['message']}"
                    ),
                    "can_retry": True,
                    "phases": progress_log,
                }

            # Update internal state
            self._current_mode = mode

            return {
                "success": True,
                "ready": True,
                "message": (
                    f"Switched to {mode} mode — "
                    f"{result['message']}"
                ),
                "can_retry": False,
                "phases": progress_log,
            }

        # ── Same mode — just verify services are up ──────────────────
        moveit_ok = await self._wait_for_moveit(timeout=30.0)
        if not moveit_ok:
            return {
                "success": True,
                "ready": True,
                "message": (
                    f"{mode.capitalize()} mode ready "
                    "(MoveIt not available — trajectory execution only)"
                ),
                "can_retry": False,
            }

        return {
            "success": True,
            "ready": True,
            "message": f"{mode.capitalize()} mode ready",
            "can_retry": False,
        }

    async def _rpc_run_proto_sim(self, client, request_id, params):
        """
        Full IK → Plan → Execute pipeline for every Cartesian pose
        the client sends.

        The client generates poses locally for the *tool_tip_link*
        (where the camera / iPhone is) and transforms them into the
        reference frame returned by get_environment_info (currently
        ``world``).  Each pose is:
            {position: [x,y,z], orientation: [qx,qy,qz,qw]}
        """
        robot_name = params.get("robot_name", list(ROBOT_CONFIG.keys())[0])
        poses = params.get("poses", [])
        idle_time = params.get("idle_time", 2.0)
        mode = params.get("mode", "simulation")
        # Cap speed for safety, matching original server behaviour.
        requested_speed = params.get("move_speed", 0.3)
        if mode in ("real", "both"):
            move_speed = min(0.3, requested_speed)
        else:
            move_speed = min(0.5, requested_speed)

        if robot_name not in ROBOT_CONFIG:
            await self._send_error(
                client, request_id, f"Unknown robot: {robot_name}"
            )
            return

        # Wait for MoveIt
        moveit_ok = await self._wait_for_moveit(timeout=30.0)
        if not moveit_ok:
            await self._send_error(
                client, request_id, "MoveIt services not available"
            )
            return

        self._proto_sim_running = True
        self._proto_sim_stop = False

        cfg = ROBOT_CONFIG[robot_name]
        total = len(poses)
        completed = 0
        ik_failed = 0
        plan_failed = 0

        self.get_logger().info(
            f"Proto-sim START: {total} poses on {robot_name} "
            f"(mode={mode}, speed={move_speed})"
        )

        # ── Move to HOME before starting protocol ──────────────────
        # The original server always starts from a known home position
        # to ensure deterministic IK seeds and consistent trajectories.
        home = cfg["home_position"]
        current = self._robot_states[robot_name].joint_positions
        if current and len(current) == 6:
            at_home = all(
                abs(current[i] - home[i]) < 0.05 for i in range(6)
            )
        else:
            at_home = False

        if not at_home:
            self.get_logger().info(
                f"Moving {robot_name} to HOME before protocol…"
            )
            home_ok = await self._move_to_home(robot_name)
            if not home_ok:
                self.get_logger().warn(
                    "Could not reach home, starting from current position"
                )
            else:
                self.get_logger().info(
                    f"{robot_name} at HOME, starting protocol"
                )
        else:
            self.get_logger().info(
                f"{robot_name} already at HOME, starting protocol"
            )

        for i, pose_data in enumerate(poses):
            if self._proto_sim_stop:
                self.get_logger().info("Proto-sim stopped by user")
                break

            pose_name = pose_data.get("name", f"pose_{i}")
            position = pose_data.get("position", [0, 0, 0])
            orientation = pose_data.get("orientation", [0, 0, 0, 1])

            # Send progress feedback
            await client.websocket.send(json.dumps({
                "type": "rpc_feedback",
                "request_id": request_id,
                "current_pose_index": i,
                "total_poses": total,
                "current_pose_name": pose_name,
                "progress_percent": (i / total) * 100 if total > 0 else 0,
                "status": "moving",
            }))

            self.get_logger().info(
                f"  [{i+1}/{total}] {pose_name} → "
                f"pos=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f}) "
                f"quat=({orientation[0]:.4f},{orientation[1]:.4f},{orientation[2]:.4f},{orientation[3]:.4f})"
            )

            result = await self._move_to_pose(
                robot_name, position, orientation,
                velocity_scaling=move_speed,
            )

            if result == "success":
                completed += 1
                self.get_logger().info(
                    f"  [{i+1}/{total}] ✓ reached {pose_name}"
                )
                await asyncio.sleep(idle_time)
            elif result == "ik_failed":
                ik_failed += 1
                self.get_logger().warn(
                    f"  [{i+1}/{total}] ✗ IK failed for {pose_name}"
                )
            elif result == "plan_failed":
                plan_failed += 1
                self.get_logger().warn(
                    f"  [{i+1}/{total}] ✗ Planning failed for {pose_name}"
                )
            else:
                plan_failed += 1
                self.get_logger().warn(
                    f"  [{i+1}/{total}] ✗ Execution failed for {pose_name}"
                )

        # Return home
        if completed > 0 and not self._proto_sim_stop:
            self.get_logger().info(f"Returning {robot_name} to home…")
            await self._move_to_home(robot_name)

        self._proto_sim_running = False

        self.get_logger().info(
            f"Proto-sim DONE: {completed}/{total} ok, "
            f"{ik_failed} IK-fail, {plan_failed} plan-fail"
        )

        await client.websocket.send(json.dumps({
            "type": "rpc_result",
            "request_id": request_id,
            "success": True,
            "message": (
                f"Completed {completed}/{total} poses "
                f"({ik_failed} IK failed, {plan_failed} plan failed)"
            ),
            "completed": completed,
            "collision_rejected": 0,
            "ik_failed": ik_failed,
            "real_failed": plan_failed,
            "total": total,
            "stopped": self._proto_sim_stop,
        }))

    async def _rpc_plan_motion(self, params):
        rn = params.get("robot_name", list(ROBOT_CONFIG.keys())[0])
        target = params.get("target_joints")
        if not target or len(target) != 6:
            return {"success": False, "error": "target_joints with 6 values required"}
        traj = await self._plan_to_joints(rn, target)
        if traj is None:
            return {"success": False, "error": "Planning failed"}
        return {
            "success": True,
            "waypoints": len(traj.joint_trajectory.points),
        }

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────

    async def _reply(self, client, rid, **kwargs):
        await client.websocket.send(json.dumps({
            "type": "response", "request_id": rid, **kwargs,
        }))

    async def _send_error(self, client, rid, error):
        try:
            await client.websocket.send(json.dumps({
                "type": "error", "request_id": rid, "error": error,
            }))
        except Exception as e:
            self.get_logger().error(f"Failed to send error: {e}")


# ── Entry point ──────────────────────────────────────────────────────

async def main():
    rclpy.init()

    node = CommandGatewayNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # Start websocket server in asyncio
    await node.start_websocket_server()

    # Spin ros2 executor in a background thread
    loop = asyncio.get_event_loop()
    ros_task = loop.run_in_executor(None, executor.spin)

    try:
        await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
