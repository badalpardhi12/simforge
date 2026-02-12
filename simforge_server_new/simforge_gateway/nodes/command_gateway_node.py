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

from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetCartesianPath
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotTrajectory,
)
from shape_msgs.msg import SolidPrimitive

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
        "passthrough_controller": "nakul_passthrough_trajectory_controller",
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
        "passthrough_controller": "sahadev_passthrough_trajectory_controller",
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
        # Motion scaling — configurable at launch time.
        # These are the MoveIt planning request scaling factors that
        # multiply against the per-joint limits in joint_limits.yaml.
        # Effective speed = per-joint limit × scaling factor.
        self.declare_parameter("max_velocity_scaling", 0.2)
        self.declare_parameter("max_acceleration_scaling", 0.2)

        self.ws_port = self.get_parameter("websocket_port").value
        self.ws_host = self.get_parameter("websocket_host").value
        self.max_clients = self.get_parameter("max_clients").value
        self.max_velocity_scaling = self.get_parameter("max_velocity_scaling").value
        self.max_acceleration_scaling = self.get_parameter("max_acceleration_scaling").value

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
        self._exec_fail_count = 0

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
        # On real hardware the passthrough controller forwards the
        # entire trajectory to the UR robot's internal interpolator
        # for smooth cubic/quintic spline execution.  In simulation
        # (fake hardware) the passthrough controller is not available
        # so we fall back to the scaled joint trajectory controller.
        self._traj_clients: Dict[str, ActionClient] = {}
        self._rebuild_traj_clients()

        # ── MoveIt service clients ───────────────────────────────────
        self._ik_client = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=self.cb_group
        )
        self._plan_client = self.create_client(
            GetMotionPlan, "/plan_kinematic_path", callback_group=self.cb_group
        )
        self._cartesian_path_client = self.create_client(
            GetCartesianPath, "/compute_cartesian_path", callback_group=self.cb_group
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
        self.get_logger().info(
            f"Motion scaling: vel={self.max_velocity_scaling}, "
            f"accel={self.max_acceleration_scaling}"
        )

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
        """Wait until MoveIt services are reachable."""
        self.get_logger().info("Waiting for MoveIt services…")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ik_ok = self._ik_client.service_is_ready()
            plan_ok = self._plan_client.service_is_ready()
            cart_ok = self._cartesian_path_client.service_is_ready()
            if ik_ok and plan_ok and cart_ok:
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

    async def _plan_to_pose_moveit(
        self,
        robot_name: str,
        target_pose: Pose,
        velocity_scaling: Optional[float] = None,
        acceleration_scaling: Optional[float] = None,
        min_z_path: Optional[float] = None,
    ) -> Optional[RobotTrajectory]:
        """Plan to a Cartesian pose using MoveIt pose constraints.

        Unlike _plan_to_joints (which needs a pre-solved IK solution),
        this method sends PositionConstraint + OrientationConstraint
        to MoveIt so that OMPL handles IK internally with full
        collision checking.  This avoids the problem of IK solutions
        that are kinematically valid but in self-collision.

        Parameters
        ----------
        min_z_path : float, optional
            Minimum Z height (in base_link frame) that the ik_tip_link
            must stay above during the ENTIRE trajectory — not just at
            the goal.  This is implemented as a MoveIt *path constraint*
            (a large box above that height).  Prevents OMPL from planning
            paths where the arm dips toward the table.

        Returns the RAW MoveIt trajectory (no retiming) for use in
        multi-waypoint concatenation.
        """
        if velocity_scaling is None:
            velocity_scaling = self.max_velocity_scaling
        if acceleration_scaling is None:
            acceleration_scaling = self.max_acceleration_scaling

        cfg = ROBOT_CONFIG[robot_name]

        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = cfg["planning_group"]
        mp.num_planning_attempts = 25
        mp.allowed_planning_time = 15.0
        mp.max_velocity_scaling_factor = velocity_scaling
        mp.max_acceleration_scaling_factor = acceleration_scaling

        # Workspace bounds — restrict min Z to above the table surface.
        # Table top is at z ≈ -0.03 in base_link.  We set min Z to 0.0
        # to prevent the planner from considering configurations where
        # the end-effector dips below the base.
        mp.workspace_parameters.header.frame_id = cfg["base_link"]
        mp.workspace_parameters.min_corner.x = -1.5
        mp.workspace_parameters.min_corner.y = -1.5
        mp.workspace_parameters.min_corner.z = 0.0
        mp.workspace_parameters.max_corner.x = 1.5
        mp.workspace_parameters.max_corner.y = 1.5
        mp.workspace_parameters.max_corner.z = 2.0

        # Start state = current joints
        current = self._robot_states[robot_name].joint_positions
        if current and len(current) == 6:
            mp.start_state.joint_state.name = list(cfg["joints"])
            mp.start_state.joint_state.position = list(current)
            mp.start_state.is_diff = False

        # ── Path constraints (applied to ENTIRE trajectory) ──────
        # Keep the tool tip above a minimum Z height during motion.
        # This prevents OMPL from planning paths where the arm swings
        # low — e.g. elbow dipping toward the table — even if such
        # paths are technically collision-free per the model.
        if min_z_path is not None:
            path_constraints = Constraints()

            pc_path = PositionConstraint()
            pc_path.header.frame_id = cfg["base_link"]
            pc_path.link_name = cfg["ik_tip_link"]
            pc_path.weight = 1.0

            # Create a large box that represents "allowed region":
            # X: -2 to +2, Y: -2 to +2, Z: min_z_path to +3
            # The tool tip must stay inside this box at all times.
            path_vol = BoundingVolume()
            path_box = SolidPrimitive()
            path_box.type = SolidPrimitive.BOX
            path_box.dimensions = [4.0, 4.0, 3.0 - min_z_path]  # x, y, z size
            path_vol.primitives.append(path_box)

            # Box center
            box_center = Pose()
            box_center.position.x = 0.0
            box_center.position.y = 0.0
            box_center.position.z = min_z_path + (3.0 - min_z_path) / 2.0
            box_center.orientation.w = 1.0
            path_vol.primitive_poses.append(box_center)

            pc_path.constraint_region = path_vol
            path_constraints.position_constraints.append(pc_path)
            mp.path_constraints = path_constraints

            self.get_logger().info(
                f"Path constraint: {cfg['ik_tip_link']} must stay above "
                f"z={min_z_path:.2f} in {cfg['base_link']}"
            )

        # Goal constraints = Cartesian pose (position + orientation)
        # MoveIt will handle IK internally with collision checking.
        constraints = Constraints()

        # Position constraint: small sphere around target position
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = cfg["base_link"]
        pos_constraint.link_name = cfg["ik_tip_link"]
        pos_constraint.weight = 1.0

        # Define a small bounding sphere around the target position
        bounding_vol = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]  # 1cm radius tolerance
        bounding_vol.primitives.append(sphere)

        target_pose_stamped = PoseStamped()
        target_pose_stamped.header.frame_id = cfg["base_link"]
        target_pose_stamped.pose.position = target_pose.position
        target_pose_stamped.pose.orientation.w = 1.0  # identity for the volume
        bounding_vol.primitive_poses.append(target_pose_stamped.pose)

        pos_constraint.constraint_region = bounding_vol
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0
        constraints.position_constraints.append(pos_constraint)

        # Orientation constraint
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = cfg["base_link"]
        orient_constraint.link_name = cfg["ik_tip_link"]
        orient_constraint.orientation = target_pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.05  # ~3 degrees
        orient_constraint.absolute_y_axis_tolerance = 0.05
        orient_constraint.absolute_z_axis_tolerance = 0.05
        orient_constraint.weight = 1.0
        constraints.orientation_constraints.append(orient_constraint)

        mp.goal_constraints.append(constraints)

        try:
            future = self._plan_client.call_async(req)
            result = await await_ros_future(future, timeout=30.0)
            if result.motion_plan_response.error_code.val == 1:  # SUCCESS
                pts = result.motion_plan_response.trajectory.joint_trajectory.points
                if pts:
                    last_pt = pts[-1]
                    traj_dur = (
                        last_pt.time_from_start.sec
                        + last_pt.time_from_start.nanosec * 1e-9
                    )
                    # Log start/end joints for debugging path quality
                    start_j = [f"{v:.3f}" for v in pts[0].positions]
                    end_j = [f"{v:.3f}" for v in last_pt.positions]
                    self.get_logger().info(
                        f"Pose plan for {robot_name}: {len(pts)} waypoints, "
                        f"duration={traj_dur:.2f}s, "
                        f"start_joints=[{', '.join(start_j)}], "
                        f"end_joints=[{', '.join(end_j)}]"
                    )
                # Return RAW trajectory (no retiming — caller will retime
                # after concatenation)
                return result.motion_plan_response.trajectory
            else:
                err_code = result.motion_plan_response.error_code.val
                self.get_logger().warn(
                    f"Pose planning failed for {robot_name}: "
                    f"error_code={err_code}"
                )
                return None
        except TimeoutError:
            self.get_logger().error(
                f"Pose planning service timed out for {robot_name}"
            )
            return None
        except Exception as exc:
            self.get_logger().error(f"Pose planning failed: {exc}")
            return None

    async def _plan_to_joints(
        self,
        robot_name: str,
        target_joints: List[float],
        velocity_scaling: Optional[float] = None,
        acceleration_scaling: Optional[float] = None,
        skip_retiming: bool = False,
    ) -> Optional[RobotTrajectory]:
        """Call /plan_kinematic_path and return RobotTrajectory or None.

        velocity_scaling / acceleration_scaling default to the ROS parameters
        max_velocity_scaling / max_acceleration_scaling when not provided.

        If skip_retiming is True, the raw MoveIt/TOTG trajectory is
        returned without quintic C2 retiming.  Used by multi-waypoint
        planning which needs raw position waypoints and applies a
        single retiming pass over the entire concatenated trajectory.
        """
        if velocity_scaling is None:
            velocity_scaling = self.max_velocity_scaling
        if acceleration_scaling is None:
            acceleration_scaling = self.max_acceleration_scaling

        cfg = ROBOT_CONFIG[robot_name]

        req = GetMotionPlan.Request()
        mp = req.motion_plan_request

        mp.group_name = cfg["planning_group"]
        mp.num_planning_attempts = 5
        mp.allowed_planning_time = 5.0
        mp.max_velocity_scaling_factor = velocity_scaling
        mp.max_acceleration_scaling_factor = acceleration_scaling

        # Workspace bounds — restrict min Z to above the table surface
        mp.workspace_parameters.header.frame_id = cfg["base_link"]
        mp.workspace_parameters.min_corner.x = -1.5
        mp.workspace_parameters.min_corner.y = -1.5
        mp.workspace_parameters.min_corner.z = 0.0
        mp.workspace_parameters.max_corner.x = 1.5
        mp.workspace_parameters.max_corner.y = 1.5
        mp.workspace_parameters.max_corner.z = 2.0

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
                # Log trajectory duration and max velocity for diagnostics
                if pts:
                    last_pt = pts[-1]
                    traj_dur = (
                        last_pt.time_from_start.sec
                        + last_pt.time_from_start.nanosec * 1e-9
                    )
                    # Compute max joint velocity across all segments
                    max_vel = 0.0
                    for k in range(1, len(pts)):
                        dt = (
                            (pts[k].time_from_start.sec + pts[k].time_from_start.nanosec * 1e-9)
                            - (pts[k-1].time_from_start.sec + pts[k-1].time_from_start.nanosec * 1e-9)
                        )
                        if dt > 0:
                            for j in range(min(len(pts[k].positions), len(pts[k-1].positions))):
                                v = abs(pts[k].positions[j] - pts[k-1].positions[j]) / dt
                                if v > max_vel:
                                    max_vel = v
                    self.get_logger().info(
                        f"Motion plan for {robot_name}: {len(pts)} waypoints, "
                        f"duration={traj_dur:.2f}s, max_seg_vel={max_vel:.3f} rad/s"
                    )
                else:
                    self.get_logger().info(
                        f"Motion plan for {robot_name}: 0 waypoints"
                    )

                if skip_retiming:
                    # Return raw MoveIt/TOTG trajectory without quintic
                    # retiming — caller will retime after concatenation.
                    return result.motion_plan_response.trajectory

                # ── Retime to enforce uniform max joint velocity ────
                # TOTG may produce slightly different peak velocities
                # per segment depending on which joints dominate each
                # segment.  Retiming caps every segment so the fastest
                # joint never exceeds velocity_scaling × per-joint limit
                # (with 10 % headroom for spline interpolation safety).
                retimed = self._retime_trajectory_constant_speed(
                    result.motion_plan_response.trajectory,
                    max_joint_vel=velocity_scaling * 1.0,  # uniform limit
                )
                return retimed
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

    def _retime_trajectory_constant_speed(
        self,
        trajectory: RobotTrajectory,
        max_joint_vel: float,
        force_zero_endpoints: bool = True,
    ) -> RobotTrajectory:
        """Retime a trajectory with uniform resampling for butter-smooth motion.

        WHY UNIFORM RESAMPLING?
        ──────────────────────
        MoveIt's Cartesian planner produces waypoints at fixed Cartesian
        step sizes (e.g. 5mm).  In joint space this translates to wildly
        varying angular increments (depending on the Jacobian).  If we
        just velocity-cap each segment individually, the resulting dt
        values span a huge range (e.g. 0.05s – 2.0s).  Fitting quintic
        splines on such non-uniform knot spacing creates Runge-like
        oscillation and the controller physically "jerks" at every
        waypoint boundary.

        The fix: after computing the total traversal time, we resample
        the entire trajectory to UNIFORM time spacing (every resample_dt
        seconds).  Then we compute velocities and accelerations from
        the uniformly-spaced positions via central finite differences.
        The result: perfectly even waypoint spacing → clean quintic
        spline interpolation → smooth robot motion.

        PIPELINE:
        1. Compute arc-length s(k) along the joint-space path.
        2. Compute total traversal time T from max_joint_vel.
        3. Create a smooth s(t) mapping (trapezoidal velocity profile
           with cosine blending at start/end for C2 continuity).
        4. Resample positions q(t) at uniform dt intervals using
           linear interpolation along the path parameterised by s.
        5. Compute velocities via central finite differences on the
           uniformly-spaced grid.
        6. Compute accelerations via second-order central differences.
        7. Clamp endpoints to zero vel+accel for smooth ramp-up/down.
        """
        import math
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        pts = list(trajectory.joint_trajectory.points)
        if len(pts) < 2:
            return trajectory

        n_joints = len(pts[0].positions)
        safe_vel = max_joint_vel * 0.85  # 15% headroom

        # ── Step 1: Compute arc-length along path ───────────────────
        # s[k] = cumulative max-joint displacement from pt[0] to pt[k]
        arc_lengths = [0.0]
        for k in range(1, len(pts)):
            max_dq = 0.0
            for j in range(n_joints):
                dq = abs(pts[k].positions[j] - pts[k-1].positions[j])
                if dq > max_dq:
                    max_dq = dq
            arc_lengths.append(arc_lengths[-1] + max_dq)

        total_arc = arc_lengths[-1]
        if total_arc < 1e-8:
            # Trajectory doesn't move — return as-is
            return trajectory

        # ── Step 2: Compute total traversal time ────────────────────
        # At constant speed = safe_vel:
        #   T_cruise = total_arc / safe_vel
        # Add ramp-up and ramp-down time (cosine blend over ramp_fraction
        # of total distance at each end).  During ramp the average speed
        # is safe_vel/2, so ramp takes twice as long as cruise for the
        # same distance.
        ramp_fraction = 0.15  # 15% of path for accel, 15% for decel
        ramp_arc = total_arc * ramp_fraction
        cruise_arc = total_arc - 2 * ramp_arc
        if cruise_arc < 0:
            # Very short trajectory — all ramp, no cruise
            ramp_arc = total_arc / 2
            cruise_arc = 0.0

        # Ramp time: average speed during cosine ramp = safe_vel * 0.5
        # → t_ramp = ramp_arc / (safe_vel * 0.5)
        t_ramp = ramp_arc / (safe_vel * 0.5) if safe_vel > 0 else 1.0
        t_cruise = cruise_arc / safe_vel if safe_vel > 0 else 0.0
        total_time = t_ramp + t_cruise + t_ramp  # accel + cruise + decel
        total_time = max(total_time, 0.5)  # minimum 0.5s trajectory

        self.get_logger().info(
            f"Retime: arc={total_arc:.4f} rad, "
            f"T={total_time:.2f}s (ramp={t_ramp:.2f}+cruise={t_cruise:.2f}+ramp={t_ramp:.2f}), "
            f"safe_vel={safe_vel:.4f} rad/s"
        )

        # ── Step 3: s(t) mapping with cosine-blended velocity profile ─
        # This gives C2 continuous speed profile:
        #   t ∈ [0, t_ramp]:         speed ramps up via cosine blend
        #   t ∈ [t_ramp, T-t_ramp]:  constant speed = safe_vel
        #   t ∈ [T-t_ramp, T]:       speed ramps down via cosine blend
        def s_of_t(t):
            """Map time → arc-length with smooth acceleration profile."""
            if t <= 0:
                return 0.0
            if t >= total_time:
                return total_arc

            t1 = t_ramp           # end of acceleration
            t2 = t_ramp + t_cruise  # start of deceleration
            t3 = total_time       # end

            if t <= t1 and t_ramp > 0:
                # Cosine ramp-up: speed = safe_vel * 0.5 * (1 - cos(π * t / t_ramp))
                # Integral: s = safe_vel * 0.5 * (t - (t_ramp/π) * sin(π * t / t_ramp))
                phase = math.pi * t / t_ramp
                s = safe_vel * 0.5 * (t - (t_ramp / math.pi) * math.sin(phase))
                return s
            elif t <= t2:
                # Cruise at safe_vel
                s_at_t1 = ramp_arc  # integral of ramp-up
                s = s_at_t1 + safe_vel * (t - t1)
                return s
            elif t_ramp > 0:
                # Cosine ramp-down
                s_at_t2 = ramp_arc + cruise_arc
                t_local = t - t2
                phase = math.pi * t_local / t_ramp
                s = s_at_t2 + safe_vel * 0.5 * (t_local - (t_ramp / math.pi) * math.sin(phase))
                return s
            else:
                # No ramp — constant speed throughout
                return safe_vel * t

        # ── Step 4: Resample to uniform time spacing ────────────────
        # Choose resample_dt to give a reasonable number of points.
        # For the UR driver at 500Hz, points every 0.1-0.2s is ideal
        # (the controller interpolates quintic between them at 500Hz).
        resample_dt = 0.08  # 80ms → 12.5 pts/sec, good for quintic splines
        n_resampled = max(int(total_time / resample_dt) + 1, 3)
        uniform_dt = total_time / (n_resampled - 1)

        # Build position lookup: given arc-length s, find interpolated
        # joint positions along the original path.
        def interp_position_at_s(s_target):
            """Linearly interpolate joint positions at arc-length s_target."""
            if s_target <= 0:
                return list(pts[0].positions)
            if s_target >= total_arc:
                return list(pts[-1].positions)

            # Binary search for the segment containing s_target
            lo, hi = 0, len(arc_lengths) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if arc_lengths[mid] <= s_target:
                    lo = mid
                else:
                    hi = mid

            # Linear interpolation within segment [lo, hi]
            ds = arc_lengths[hi] - arc_lengths[lo]
            if ds < 1e-12:
                return list(pts[lo].positions)
            alpha = (s_target - arc_lengths[lo]) / ds

            result = []
            for j in range(n_joints):
                q = pts[lo].positions[j] + alpha * (pts[hi].positions[j] - pts[lo].positions[j])
                result.append(q)
            return result

        # Generate uniformly-spaced points
        new_points = []
        for i in range(n_resampled):
            t = i * uniform_dt
            s = s_of_t(t)
            pos = interp_position_at_s(s)

            pt = JointTrajectoryPoint()
            pt.positions = pos
            sec = int(t)
            nsec = int((t - sec) * 1e9)
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            new_points.append(pt)

        # ── Step 5: Compute velocities (central differences) ────────
        # On uniformly-spaced grid, central differences are accurate
        # and produce smooth velocity profiles.
        for k in range(len(new_points)):
            vels = []
            for j in range(n_joints):
                if k == 0 or k == len(new_points) - 1:
                    vels.append(0.0)
                else:
                    # Central difference: v = (q[k+1] - q[k-1]) / (2 * dt)
                    dq = new_points[k+1].positions[j] - new_points[k-1].positions[j]
                    v = dq / (2 * uniform_dt)
                    v = max(-safe_vel, min(safe_vel, v))
                    vels.append(v)
            new_points[k].velocities = vels

        # ── Step 6: Compute accelerations (second-order central diff) ─
        for k in range(len(new_points)):
            accels = []
            for j in range(n_joints):
                if k == 0 or k == len(new_points) - 1:
                    accels.append(0.0)
                elif k == 1:
                    # Forward difference of velocity
                    a = (new_points[k+1].velocities[j] - new_points[k].velocities[j]) / uniform_dt
                    accels.append(a)
                elif k == len(new_points) - 2:
                    # Backward difference of velocity
                    a = (new_points[k].velocities[j] - new_points[k-1].velocities[j]) / uniform_dt
                    accels.append(a)
                else:
                    # Central second difference: a = (q[k+1] - 2*q[k] + q[k-1]) / dt²
                    a = (new_points[k+1].positions[j] - 2*new_points[k].positions[j] + new_points[k-1].positions[j]) / (uniform_dt ** 2)
                    accels.append(a)
            new_points[k].accelerations = accels

        trajectory.joint_trajectory.points = new_points

        # ── Diagnostics ─────────────────────────────────────────────
        actual_max_v = 0.0
        for k in range(len(new_points)):
            for j in range(n_joints):
                v = abs(new_points[k].velocities[j])
                if v > actual_max_v:
                    actual_max_v = v

        # Compute jerk stats
        max_jerk = 0.0
        for k in range(1, len(new_points)):
            for j in range(n_joints):
                prev_a = new_points[k-1].accelerations[j] if new_points[k-1].accelerations else 0.0
                curr_a = new_points[k].accelerations[j] if new_points[k].accelerations else 0.0
                jerk = abs(curr_a - prev_a) / uniform_dt
                if jerk > max_jerk:
                    max_jerk = jerk

        self.get_logger().info(
            f"Retimed trajectory: {len(new_points)} pts, "
            f"duration={total_time:.2f}s, "
            f"uniform_dt={uniform_dt:.4f}s, "
            f"max_vel_cap={safe_vel:.4f} rad/s, "
            f"actual_max_vel={actual_max_v:.4f} rad/s, "
            f"max_jerk={max_jerk:.3f} rad/s³, "
            f"interp=quintic(C2)+uniform_resample"
        )

        # Dump first/last few points for quick sanity check
        for k in [0, 1, 2, len(new_points)//2, len(new_points)-3, len(new_points)-2, len(new_points)-1]:
            if 0 <= k < len(new_points):
                t = new_points[k].time_from_start.sec + new_points[k].time_from_start.nanosec * 1e-9
                mv = max(abs(v) for v in new_points[k].velocities) if new_points[k].velocities else 0
                ma = max(abs(a) for a in new_points[k].accelerations) if new_points[k].accelerations else 0
                self.get_logger().info(
                    f"  sample pt[{k:03d}] t={t:.3f}s |v|={mv:.5f} |a|={ma:.5f}"
                )

        return trajectory

    # ─────────────────────────────────────────────────────────────────
    # Multi-waypoint planning — single continuous trajectory
    # ─────────────────────────────────────────────────────────────────

    async def _plan_cartesian_path(
        self,
        robot_name: str,
        waypoints: list,
        velocity_scaling: Optional[float] = None,
        acceleration_scaling: Optional[float] = None,
        max_step: float = 0.01,
    ) -> Optional[RobotTrajectory]:
        """Plan a single trajectory through multiple Cartesian waypoints.

        Uses MoveIt's /compute_cartesian_path service which generates a
        single RobotTrajectory that moves the end-effector through all
        waypoints in sequence with smooth interpolation.

        Parameters
        ----------
        robot_name : str
            Robot to plan for.
        waypoints : list of (position, orientation) tuples
            Each waypoint is ([x,y,z], [qx,qy,qz,qw]).
        velocity_scaling : float, optional
            MoveIt velocity scaling (0, 1].
        acceleration_scaling : float, optional
            MoveIt acceleration scaling (0, 1].
        max_step : float
            Maximum Cartesian distance between interpolated points (m).
            Smaller = more points = smoother but slower to plan.

        Returns
        -------
        RobotTrajectory or None
            Single trajectory through all waypoints, or None on failure.
        """
        if velocity_scaling is None:
            velocity_scaling = self.max_velocity_scaling
        if acceleration_scaling is None:
            acceleration_scaling = self.max_acceleration_scaling

        cfg = ROBOT_CONFIG[robot_name]

        req = GetCartesianPath.Request()
        req.header.frame_id = cfg["base_link"]
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = cfg["planning_group"]
        req.link_name = cfg["ik_tip_link"]

        # Start state = current joints
        current = self._robot_states[robot_name].joint_positions
        if current and len(current) == 6:
            req.start_state.joint_state.name = list(cfg["joints"])
            req.start_state.joint_state.position = list(current)
            req.start_state.is_diff = False

        # Build waypoint poses
        for pos, orient in waypoints:
            p = Pose()
            p.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            p.orientation = Quaternion(
                x=float(orient[0]), y=float(orient[1]),
                z=float(orient[2]), w=float(orient[3]),
            )
            req.waypoints.append(p)

        req.max_step = max_step
        # Disable jump detection — poses are pre-validated by IK
        req.jump_threshold = 0.0
        req.avoid_collisions = True

        try:
            future = self._cartesian_path_client.call_async(req)
            result = await await_ros_future(future, timeout=30.0)

            fraction = result.fraction
            self.get_logger().info(
                f"Cartesian path for {robot_name}: "
                f"fraction={fraction:.2%} through {len(waypoints)} waypoints"
            )

            if fraction < 0.95:
                self.get_logger().warn(
                    f"Cartesian path only achieved {fraction:.1%} — "
                    f"falling back to joint-space multi-waypoint planning"
                )
                return None

            # The Cartesian path service returns a time-parameterized
            # trajectory.  Apply our retiming for velocity capping.
            max_vel = (velocity_scaling or self.max_velocity_scaling) * 1.0
            retimed = self._retime_trajectory_constant_speed(
                result.solution, max_joint_vel=max_vel,
            )
            return retimed

        except TimeoutError:
            self.get_logger().error(
                f"Cartesian path service timed out for {robot_name}"
            )
            return None
        except Exception as exc:
            self.get_logger().error(
                f"Cartesian path service failed: {exc}"
            )
            return None

    async def _plan_multi_waypoint_trajectory(
        self,
        robot_name: str,
        poses: list,
        velocity_scaling: Optional[float] = None,
        idle_time: float = 0.0,
    ) -> Optional[tuple]:
        """Plan a single collision-free trajectory through multiple poses.

        Strategy (Cartesian-first, OMPL-fallback):
        1. Try Cartesian path planning (straight-line in workspace) for
           each segment.  This produces safe, predictable motions with
           no wild joint excursions.
        2. If Cartesian planning fails for a segment (e.g. collision or
           singularity), fall back to OMPL pose-based planning for
           that segment only.
        3. Concatenates all segments into ONE continuous trajectory.
        4. If idle_time > 0, inserts dwell periods at each waypoint.
        5. Applies a SINGLE retiming pass for velocity capping and
           quintic C2 spline preparation across ALL segment boundaries.

        Returns
        -------
        (RobotTrajectory, list of (pose_index, time_from_start), valid_indices)
            Or None if planning fails.
        """
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration

        cfg = ROBOT_CONFIG[robot_name]
        if velocity_scaling is None:
            velocity_scaling = self.max_velocity_scaling

        current = self._robot_states[robot_name].joint_positions

        # ── Plan collision-free segments (Cartesian first, OMPL fallback) ──
        segments = []
        valid_indices = []
        failed_names = []

        for i, pose_data in enumerate(poses):
            pose_name = pose_data.get("name", f"pose_{i}")
            position = pose_data.get("position", [0, 0, 0])
            orientation = pose_data.get("orientation", [0, 0, 0, 1])

            self.get_logger().info(
                f"Planning segment {len(segments)+1} to pose {i} ({pose_name}): "
                f"pos=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}], "
                f"orn=[{orientation[0]:.3f}, {orientation[1]:.3f}, {orientation[2]:.3f}, {orientation[3]:.3f}]"
            )

            # ── Attempt 1: Cartesian path (straight-line in workspace) ──
            # This is the safest approach: the tool tip moves in a
            # straight line from the current position to the goal.
            # No wild joint excursions.
            seg_traj = None
            try:
                cart_req = GetCartesianPath.Request()
                cart_req.header.frame_id = cfg["base_link"]
                cart_req.header.stamp = self.get_clock().now().to_msg()
                cart_req.group_name = cfg["planning_group"]
                cart_req.link_name = cfg["ik_tip_link"]

                # Start state = current joints
                cur_joints = self._robot_states[robot_name].joint_positions
                if cur_joints and len(cur_joints) == 6:
                    cart_req.start_state.joint_state.name = list(cfg["joints"])
                    cart_req.start_state.joint_state.position = list(cur_joints)
                    cart_req.start_state.is_diff = False

                # Single waypoint = the target pose
                target = Pose()
                target.position = Point(
                    x=float(position[0]), y=float(position[1]),
                    z=float(position[2]),
                )
                target.orientation = Quaternion(
                    x=float(orientation[0]), y=float(orientation[1]),
                    z=float(orientation[2]), w=float(orientation[3]),
                )
                cart_req.waypoints.append(target)

                cart_req.max_step = 0.005  # 5mm resolution
                cart_req.jump_threshold = 0.0  # disable jump detection
                cart_req.avoid_collisions = True

                future = self._cartesian_path_client.call_async(cart_req)
                result = await await_ros_future(future, timeout=15.0)
                fraction = result.fraction

                if fraction >= 0.98:
                    seg_traj = result.solution
                    self.get_logger().info(
                        f"Cartesian path OK for {pose_name}: "
                        f"fraction={fraction:.1%}, "
                        f"{len(seg_traj.joint_trajectory.points)} pts"
                    )
                else:
                    self.get_logger().info(
                        f"Cartesian path for {pose_name}: "
                        f"fraction={fraction:.1%} — trying OMPL"
                    )

            except Exception as exc:
                self.get_logger().info(
                    f"Cartesian path failed for {pose_name}: {exc} — trying OMPL"
                )

            # ── Attempt 2: IK + joint-space OMPL (fallback) ────────
            # Solve IK with multiple seeds, then plan to joints.
            # This is safer than pose-based OMPL which can find
            # wild wraparound joint configurations.
            if seg_traj is None:
                self.get_logger().info(
                    f"IK+joint fallback for {pose_name}, "
                    f"cur_joints=[{', '.join(f'{v:.3f}' for v in (self._robot_states[robot_name].joint_positions or []))}]"
                )
                pose_obj = Pose()
                pose_obj.position = Point(
                    x=float(position[0]), y=float(position[1]),
                    z=float(position[2]),
                )
                pose_obj.orientation = Quaternion(
                    x=float(orientation[0]), y=float(orientation[1]),
                    z=float(orientation[2]), w=float(orientation[3]),
                )

                cur_joints = self._robot_states[robot_name].joint_positions
                seeds = [
                    cur_joints if cur_joints and len(cur_joints) == 6 else None,
                    list(cfg["home_position"]),
                    None,  # Let IK solver pick random seed
                ]

                for seed_idx, seed in enumerate(seeds):
                    ik_result = await self._solve_ik(
                        robot_name, pose_obj, seed_joints=seed
                    )
                    if ik_result is None:
                        continue

                    # Validate: check joint displacement from current
                    if cur_joints and len(cur_joints) == 6:
                        max_disp = max(
                            abs(ik_result[j] - cur_joints[j]) for j in range(6)
                        )
                        if max_disp > 3.14:  # > 180° on any joint → skip
                            self.get_logger().info(
                                f"IK seed {seed_idx}: max joint disp "
                                f"{max_disp:.2f} rad > 3.14 — skipping"
                            )
                            continue

                    # Plan to this IK solution
                    seg_traj = await self._plan_to_joints(
                        robot_name, ik_result,
                        velocity_scaling=velocity_scaling,
                        skip_retiming=True,
                    )
                    if seg_traj is not None:
                        self.get_logger().info(
                            f"IK+OMPL OK for {pose_name} (seed {seed_idx})"
                        )
                        break
                    else:
                        self.get_logger().info(
                            f"IK seed {seed_idx} plan failed for {pose_name}"
                        )

            if seg_traj is None:
                failed_names.append(pose_name)
                self.get_logger().warn(
                    f"All planning failed for waypoint {i} ({pose_name}) — skipping"
                )
                continue

            segments.append(seg_traj)
            valid_indices.append(i)

            # Log segment joint displacement for safety diagnostics
            final_pts = seg_traj.joint_trajectory.points
            if final_pts:
                start_j = list(final_pts[0].positions)
                end_j = list(final_pts[-1].positions)
                max_disp = max(abs(end_j[j] - start_j[j]) for j in range(6))
                self.get_logger().info(
                    f"Segment {len(segments)} ({pose_name}): "
                    f"{len(final_pts)} pts, max_joint_disp={max_disp:.3f} rad, "
                    f"end_joints=[{', '.join(f'{v:.3f}' for v in end_j)}]"
                )

            # Update start state for next segment
            final_pts = seg_traj.joint_trajectory.points
            if final_pts:
                self._robot_states[robot_name].joint_positions = list(
                    final_pts[-1].positions
                )

        # Restore actual joint state
        if current and len(current) == 6:
            self._robot_states[robot_name].joint_positions = current

        if not segments:
            self.get_logger().error("All pose plans failed — cannot build trajectory")
            return None

        if failed_names:
            self.get_logger().warn(
                f"Planning failed for {len(failed_names)} poses: "
                f"{failed_names[:5]}{'...' if len(failed_names) > 5 else ''}"
            )

        self.get_logger().info(
            f"All {len(segments)} segments planned "
            f"(Cartesian-first with collision checking)"
        )

        # ── Phase 3: Concatenate segments into one trajectory ────────
        # CRITICAL: Extract ONLY positions from each raw MoveIt segment.
        # Discard TOTG velocities/accelerations/timing because they
        # include per-segment deceleration-to-zero that would create
        # velocity discontinuities at every segment boundary.
        #
        # We assign simple placeholder timing (equispaced) here —
        # the single retiming pass in Phase 4 will recompute proper
        # timing, velocities (central differences), and accelerations
        # (second-order differences) across ALL segment boundaries
        # for a smooth quintic C2 trajectory.
        n_joints = len(cfg["joints"])
        all_positions = []    # list of position tuples
        pose_point_indices = []  # (pose_index, point_index_in_all_positions)

        for seg_idx, seg_traj in enumerate(segments):
            seg_pts = seg_traj.joint_trajectory.points
            if not seg_pts:
                continue

            # Skip the first point of each segment after the first
            # (it duplicates the last point of the previous segment)
            start_idx = 1 if seg_idx > 0 and all_positions else 0

            for pt_idx in range(start_idx, len(seg_pts)):
                all_positions.append(list(seg_pts[pt_idx].positions))

            # Record the point index where this pose's waypoint lands
            pose_point_indices.append(
                (valid_indices[seg_idx], len(all_positions) - 1)
            )

            # Insert dwell point if idle_time > 0 (pause at each pose)
            # The dwell uses the same position — retiming will assign
            # zero velocity/acceleration because dq=0.
            is_last = (seg_idx == len(segments) - 1)
            if idle_time > 0 and not is_last:
                all_positions.append(list(seg_pts[-1].positions))

        if not all_positions:
            self.get_logger().error("No trajectory points after concatenation")
            return None

        # Build trajectory with ONLY positions + placeholder timing.
        # Use equispaced 0.1s intervals — the retiming pass will
        # adjust timing based on actual joint displacements.
        all_points = []
        for k, pos in enumerate(all_positions):
            pt = JointTrajectoryPoint()
            pt.positions = pos
            # Placeholder timing — will be overwritten by retiming
            t = k * 0.1
            sec = int(t)
            nsec = int((t - sec) * 1e9)
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            all_points.append(pt)

        # Package into RobotTrajectory
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory = JointTrajectory()
        trajectory.joint_trajectory.joint_names = list(cfg["joints"])
        trajectory.joint_trajectory.points = all_points

        # ── Phase 4: SINGLE retiming pass across ALL segments ────────
        # Resamples to UNIFORM time spacing with cosine-blended velocity
        # profile.  Produces smooth quintic C2 trajectory with no
        # non-uniform dt artefacts.
        retimed = self._retime_trajectory_constant_speed(
            trajectory,
            max_joint_vel=velocity_scaling * 1.0,
            force_zero_endpoints=(idle_time <= 0),
        )

        # ── Compute pose_times by matching positions ─────────────────
        # After resampling, point indices changed.  Find the resampled
        # point whose positions are closest to each target pose's final
        # joint positions (the last point of each segment before concat).
        pts = retimed.joint_trajectory.points
        pose_times = []

        # Collect target joint positions for each segment endpoint
        target_joints = []
        for seg_idx, seg_traj in enumerate(segments):
            seg_pts = seg_traj.joint_trajectory.points
            if seg_pts:
                target_joints.append(
                    (valid_indices[seg_idx], list(seg_pts[-1].positions))
                )

        for pose_idx, target_j in target_joints:
            best_k = 0
            best_dist = float('inf')
            for k in range(len(pts)):
                dist = max(
                    abs(pts[k].positions[j] - target_j[j])
                    for j in range(min(len(pts[k].positions), len(target_j)))
                )
                if dist < best_dist:
                    best_dist = dist
                    best_k = k
            t = (
                pts[best_k].time_from_start.sec +
                pts[best_k].time_from_start.nanosec * 1e-9
            )
            pose_times.append((pose_idx, t))

        total_dur = (
            pts[-1].time_from_start.sec +
            pts[-1].time_from_start.nanosec * 1e-9
        ) if pts else 0.0

        self.get_logger().info(
            f"Concatenated trajectory: {len(pts)} points, "
            f"duration={total_dur:.2f}s, segments={len(segments)}, "
            f"dwells={'yes' if idle_time > 0 else 'no'} "
            f"(single continuous execution, collision-checked)"
        )

        return retimed, pose_times, valid_indices

    async def _execute_trajectory(
        self, robot_name: str, trajectory: RobotTrajectory, timeout: float = 60.0
    ) -> bool:
        """Send a RobotTrajectory to the FollowJointTrajectory action.

        Includes retry logic: if the trajectory is rejected or aborted due to
        a transient controller deactivation (e.g. caused by the other robot's
        connection dropping, which triggers controller_stopper), the method
        waits for the controller to come back and retries once.
        """
        client = self._traj_clients.get(robot_name)
        if client is None:
            self.get_logger().error(f"No trajectory client for {robot_name}")
            return False

        max_attempts = 2  # original attempt + 1 retry

        for attempt in range(max_attempts):
            # Wait for action server (non-blocking poll)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if client.server_is_ready():
                    break
                await asyncio.sleep(0.2)
            else:
                self.get_logger().error(
                    f"Trajectory action server not available for {robot_name}"
                )
                if attempt < max_attempts - 1:
                    self.get_logger().info(
                        f"Waiting for controller recovery before retry "
                        f"(attempt {attempt+1}/{max_attempts})…"
                    )
                    # Log which robots have connection issues
                    self._log_cross_robot_diagnostics(robot_name)
                    await asyncio.sleep(5.0)
                    continue
                return False

            # ── Ensure the UR robot program is running ──────────────
            ready = await self._ensure_robot_ready(robot_name, timeout=15.0)
            if not ready:
                self.get_logger().error(
                    f"Robot program not running on {robot_name} — "
                    f"cannot execute trajectory"
                )
                return False

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = trajectory.joint_trajectory
            # Allow generous goal_time_tolerance for the scaled
            # controller — speed scaling can slow execution, so
            # give it plenty of margin beyond the planned duration.
            goal.goal_time_tolerance.sec = 30
            goal.goal_time_tolerance.nanosec = 0

            # ── Timestamp sanity checks ─────────────────────────────
            # MoveIt should produce monotonically increasing timestamps
            # via AddTimeOptimalParameterization.  If timestamps are
            # zero or non-increasing, the trajectory is broken (e.g.
            # adapter failure) — refuse to execute rather than risk
            # the controller doing aggressive spline interpolation.
            pts = goal.trajectory.points

            # Strip a zero-time leading point (common TOPP artefact
            # where point 0 duplicates the start state at t=0)
            if len(pts) >= 2:
                t0 = pts[0].time_from_start.sec + pts[0].time_from_start.nanosec * 1e-9
                t1 = pts[1].time_from_start.sec + pts[1].time_from_start.nanosec * 1e-9
                if t0 >= t1 or t0 == 0.0:
                    pts = list(pts[1:])
                    goal.trajectory.points = pts

            # Reject if remaining timestamps are not strictly increasing
            if len(pts) >= 2:
                times = [
                    p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                    for p in pts
                ]
                monotonic = all(
                    times[i] < times[i + 1]
                    for i in range(len(times) - 1)
                )
                if not monotonic:
                    self.get_logger().error(
                        f"SAFETY: Rejecting trajectory for {robot_name} — "
                        f"timestamps are not strictly increasing "
                        f"(times={[f'{t:.3f}' for t in times]}). "
                        f"This usually means a MoveIt adapter failed."
                    )
                    return False

            n_pts = len(goal.trajectory.points)
            self.get_logger().info(
                f"Executing trajectory on {robot_name} ({n_pts} points)"
                + (f" [retry {attempt}]" if attempt > 0 else "")
            )

            try:
                # Send goal
                send_future = client.send_goal_async(goal)
                goal_handle = await await_ros_future(send_future, timeout=10.0)

                if not goal_handle.accepted:
                    self.get_logger().warn(
                        f"Trajectory goal rejected for {robot_name}"
                    )
                    self._log_cross_robot_diagnostics(robot_name)
                    if attempt < max_attempts - 1:
                        self.get_logger().info(
                            "Controller may have been deactivated by "
                            "controller_stopper — waiting for recovery…"
                        )
                        await asyncio.sleep(5.0)
                        continue
                    return False

                self.get_logger().info(
                    f"Trajectory accepted for {robot_name}, waiting…"
                )

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
                        f"Trajectory execution error on {robot_name}: "
                        f"code={error_code}"
                    )
                    self._log_cross_robot_diagnostics(robot_name)
                    if attempt < max_attempts - 1:
                        self.get_logger().info(
                            f"Trajectory aborted (code={error_code}) — "
                            "may be caused by cross-robot controller_stopper. "
                            "Waiting for recovery before retry…"
                        )
                        await asyncio.sleep(5.0)
                        continue
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

        return False  # Should not reach here

    def _log_cross_robot_diagnostics(self, executing_robot: str) -> None:
        """Log connection status of all robots for diagnostic purposes.

        When a trajectory fails on one robot, it may be because the other
        robot's connection dropped, causing the shared ros2_control_node
        and controller_stopper to deactivate controllers.
        """
        current_mode = self._read_current_mode()
        if current_mode not in ("real", "both"):
            return  # Diagnostics only relevant in real mode

        for rn in ROBOT_CONFIG:
            prog = self._robot_program_running.get(rn, False)
            st = self._robot_states.get(rn)
            has_joints = st and len(st.joint_positions) == 6
            age = time.time() - st.last_update if st and st.last_update > 0 else -1
            marker = " ← EXECUTING" if rn == executing_robot else ""
            if not prog:
                self.get_logger().warn(
                    f"  ⚠ {rn}: program_running=False, "
                    f"joints={'ok' if has_joints else 'MISSING'}, "
                    f"last_update={age:.1f}s ago{marker}"
                )
            else:
                self.get_logger().info(
                    f"  ✓ {rn}: program_running=True, "
                    f"joints={'ok' if has_joints else 'MISSING'}, "
                    f"last_update={age:.1f}s ago{marker}"
                )

    async def _move_to_pose(
        self,
        robot_name: str,
        position: List[float],
        orientation: List[float],
        velocity_scaling: Optional[float] = None,
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
        """Plan and execute a return-to-home motion.

        Includes recovery logic: if the controller was deactivated (e.g.
        by cross-robot controller_stopper), waits for it to come back.
        """
        cfg = ROBOT_CONFIG[robot_name]

        # Ensure the robot is ready before planning home
        # (the controller may have been deactivated by controller_stopper
        # if the OTHER robot's connection dropped)
        ready = await self._ensure_robot_ready(robot_name, timeout=15.0)
        if not ready:
            self.get_logger().warn(
                f"Robot program not running for {robot_name} — "
                f"cannot plan home. Skipping go-home."
            )
            return False

        trajectory = await self._plan_to_joints(
            robot_name, cfg["home_position"],
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

    # ── Trajectory action-client management ─────────────────────────

    def _rebuild_traj_clients(self) -> None:
        """(Re)create FollowJointTrajectory action clients.

        Always uses the ``ScaledJointTrajectoryController`` which is
        reliable in dual-robot setups.  The passthrough controller
        cannot be used because one robot's RTDE instability causes
        the other robot's controller to be deactivated mid-trajectory
        by the controller_stopper.
        """
        # Destroy old clients first (safe even if dict is empty)
        for old_client in self._traj_clients.values():
            old_client.destroy()
        self._traj_clients.clear()

        for robot_name, cfg in ROBOT_CONFIG.items():
            ctrl = cfg["controller"]
            action_ns = f"/{ctrl}/follow_joint_trajectory"
            self._traj_clients[robot_name] = ActionClient(
                self, FollowJointTrajectory, action_ns,
                callback_group=self.cb_group,
            )
            self.get_logger().info(
                f"Trajectory action client for {robot_name}: {action_ns}"
            )

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
            # Recreate action clients for the new mode's controllers
            self._rebuild_traj_clients()

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
        Unified multi-waypoint trajectory execution.

        Instead of planning and executing each pose as a separate
        trajectory (which causes stop-start jerkiness), this method:

        1. Pre-solves IK for ALL poses at once.
        2. Plans a SINGLE trajectory through all waypoints.
        3. Executes it in ONE FollowJointTrajectory action goal.
        4. Reports progress based on elapsed time vs. pose time mapping.

        If idle_time > 0, dwell periods are inserted at each waypoint
        so the robot pauses momentarily (for photo capture) but still
        executes the entire trajectory as a single continuous action
        — no re-planning or re-accelerating between poses.

        Falls back to the legacy pose-by-pose approach if unified
        planning fails.
        """
        robot_name = params.get("robot_name", list(ROBOT_CONFIG.keys())[0])
        poses = params.get("poses", [])
        idle_time = params.get("idle_time", 2.0)
        mode = params.get("mode", "simulation")
        requested_speed = params.get("move_speed", self.max_velocity_scaling)
        move_speed = min(self.max_velocity_scaling, requested_speed)

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

        self.get_logger().info(
            f"Proto-sim START: {total} poses on {robot_name} "
            f"(mode={mode}, speed={move_speed}, idle={idle_time}s)"
        )

        # ── Move to HOME before starting protocol ──────────────────
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
                self.get_logger().info(f"{robot_name} at HOME, starting protocol")
        else:
            self.get_logger().info(f"{robot_name} already at HOME")

        # ── Unified multi-waypoint planning ────────────────────────
        await client.websocket.send(json.dumps({
            "type": "rpc_feedback",
            "request_id": request_id,
            "current_pose_index": 0,
            "total_poses": total,
            "progress_percent": 0,
            "status": "planning",
            "message": f"Planning trajectory through {total} poses…",
        }))

        plan_result = await self._plan_multi_waypoint_trajectory(
            robot_name, poses,
            velocity_scaling=move_speed,
            idle_time=idle_time,
        )

        if plan_result is None:
            # Unified planning failed — fall back to legacy pose-by-pose
            self.get_logger().warn(
                "Multi-waypoint planning failed — "
                "falling back to pose-by-pose execution"
            )
            await self._rpc_run_proto_sim_legacy(
                client, request_id, robot_name, poses,
                idle_time, mode, move_speed, cfg, total,
            )
            return

        trajectory, pose_times, valid_indices = plan_result
        ik_failed = total - len(valid_indices)

        # Compute total trajectory duration for timeout
        pts = trajectory.joint_trajectory.points
        total_dur = (
            pts[-1].time_from_start.sec +
            pts[-1].time_from_start.nanosec * 1e-9
        ) if pts else 0.0

        self.get_logger().info(
            f"Executing unified trajectory: {len(pts)} points, "
            f"duration={total_dur:.2f}s, poses={len(valid_indices)}"
        )

        # ── Ensure robot program is running ────────────────────────
        ready = await self._ensure_robot_ready(robot_name, timeout=15.0)
        if not ready:
            current_mode = self._read_current_mode()
            if current_mode in ("real", "both"):
                self.get_logger().error(
                    f"Robot program not running on {robot_name}"
                )
                await self._send_error(
                    client, request_id,
                    f"Robot program not running on {robot_name}"
                )
                self._proto_sim_running = False
                return

        # ── Send progress updates during execution ─────────────────
        # Start execution and progress monitoring concurrently.
        exec_timeout = max(total_dur * 2.0, 60.0)

        # Launch execution in background
        exec_task = asyncio.ensure_future(
            self._execute_trajectory(robot_name, trajectory, timeout=exec_timeout)
        )

        # Monitor progress based on elapsed time
        exec_start = time.monotonic()
        pose_time_idx = 0  # next pose to report
        last_reported = -1

        while not exec_task.done():
            elapsed = time.monotonic() - exec_start
            progress = min(elapsed / total_dur, 1.0) if total_dur > 0 else 1.0

            # Report progress for each pose as we pass its time
            while (pose_time_idx < len(pose_times) and
                   elapsed >= pose_times[pose_time_idx][1]):
                pi = pose_times[pose_time_idx][0]
                if pi != last_reported:
                    pose_name = poses[pi].get("name", f"pose_{pi}")
                    await client.websocket.send(json.dumps({
                        "type": "rpc_feedback",
                        "request_id": request_id,
                        "current_pose_index": pi,
                        "total_poses": total,
                        "current_pose_name": pose_name,
                        "progress_percent": progress * 100,
                        "status": "reached",
                    }))
                    self.get_logger().info(
                        f"  [{pi+1}/{total}] ✓ passed {pose_name} "
                        f"(t={elapsed:.1f}s)"
                    )
                    last_reported = pi
                pose_time_idx += 1

            if self._proto_sim_stop:
                self.get_logger().info("Proto-sim stop requested — cancelling")
                # Cancel the action if possible
                break

            await asyncio.sleep(0.5)

        # Wait for execution result
        try:
            exec_ok = await asyncio.wait_for(
                asyncio.shield(exec_task), timeout=10.0
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            exec_ok = False

        completed = len(valid_indices) if exec_ok else 0
        plan_failed = 0 if exec_ok else 1

        if exec_ok:
            self.get_logger().info(
                f"Unified trajectory executed successfully: "
                f"{len(valid_indices)} poses"
            )
            self._exec_fail_count = 0
        else:
            self._exec_fail_count += 1
            self.get_logger().error(
                f"Unified trajectory execution failed"
            )
            self._log_cross_robot_diagnostics(robot_name)

        # ── Return home ────────────────────────────────────────────
        if completed > 0 and not self._proto_sim_stop:
            self.get_logger().info(f"Returning {robot_name} to home…")
            await self._move_to_home(robot_name)

        self._proto_sim_running = False

        # ── Build summary ──────────────────────────────────────────
        summary_parts = [f"Completed {completed}/{total} poses"]
        if ik_failed:
            summary_parts.append(f"{ik_failed} IK failed")
        if plan_failed:
            summary_parts.append(f"trajectory execution failed")

        hardware_issues = []
        current_mode = self._read_current_mode()
        if current_mode in ("real", "both"):
            for rn in ROBOT_CONFIG:
                if not self._robot_program_running.get(rn, False):
                    hardware_issues.append(rn)
        if hardware_issues:
            summary_parts.append(
                f"⚠ Hardware: {', '.join(hardware_issues)} disconnected"
            )

        summary = " | ".join(summary_parts)
        self.get_logger().info(
            f"Proto-sim DONE: {completed}/{total} ok, "
            f"{ik_failed} IK-fail, {plan_failed} exec-fail"
            + (f" | hw-issues: {hardware_issues}" if hardware_issues else "")
        )

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
            "stopped": self._proto_sim_stop,
            "hardware_issues": hardware_issues,
        }))

    async def _rpc_run_proto_sim_legacy(
        self, client, request_id, robot_name, poses,
        idle_time, mode, move_speed, cfg, total,
    ):
        """Legacy pose-by-pose execution fallback.

        Used when unified multi-waypoint planning fails (e.g. Cartesian
        path can't be computed, or all IK solutions fail).  Each pose
        is planned and executed as a separate trajectory.
        """
        completed = 0
        ik_failed = 0
        plan_failed = 0

        self.get_logger().info(
            "Running LEGACY pose-by-pose execution (fallback)"
        )

        for i, pose_data in enumerate(poses):
            if self._proto_sim_stop:
                self.get_logger().info("Proto-sim stopped by user")
                break

            pose_name = pose_data.get("name", f"pose_{i}")
            position = pose_data.get("position", [0, 0, 0])
            orientation = pose_data.get("orientation", [0, 0, 0, 1])

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
                f"pos=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f})"
            )

            result = await self._move_to_pose(
                robot_name, position, orientation,
                velocity_scaling=move_speed,
            )

            if result == "success":
                completed += 1
                self._exec_fail_count = 0
                self.get_logger().info(f"  [{i+1}/{total}] ✓ reached {pose_name}")
                await asyncio.sleep(idle_time)
            elif result == "ik_failed":
                ik_failed += 1
                self.get_logger().warn(f"  [{i+1}/{total}] ✗ IK failed for {pose_name}")
            elif result == "plan_failed":
                plan_failed += 1
                self.get_logger().warn(f"  [{i+1}/{total}] ✗ Planning failed for {pose_name}")
            elif result == "exec_failed":
                self._exec_fail_count += 1
                plan_failed += 1
                self._log_cross_robot_diagnostics(robot_name)
                self.get_logger().error(
                    f"  [{i+1}/{total}] ✗ Execution failed for {pose_name}"
                )
                other_robots_down = [
                    rn for rn in ROBOT_CONFIG
                    if rn != robot_name
                    and not self._robot_program_running.get(rn, False)
                ]
                diag_hint = ""
                if other_robots_down:
                    diag_hint = (
                        f" — possible cause: {', '.join(other_robots_down)} "
                        f"lost connection"
                    )
                await client.websocket.send(json.dumps({
                    "type": "rpc_feedback",
                    "request_id": request_id,
                    "current_pose_index": i,
                    "total_poses": total,
                    "current_pose_name": pose_name,
                    "progress_percent": (i / total) * 100 if total > 0 else 0,
                    "status": "exec_failed",
                    "message": f"Execution failed for {pose_name}{diag_hint}",
                }))
                if not self._robot_program_running.get(robot_name, False):
                    current_mode = self._read_current_mode()
                    if current_mode in ("real", "both"):
                        self.get_logger().error(
                            f"Robot program stopped — aborting remaining poses"
                        )
                        break
            else:
                plan_failed += 1

        self._exec_fail_count = 0

        if completed > 0 and not self._proto_sim_stop:
            self.get_logger().info(f"Returning {robot_name} to home…")
            await self._move_to_home(robot_name)

        self._proto_sim_running = False

        summary_parts = [f"Completed {completed}/{total} poses (legacy)"]
        if ik_failed:
            summary_parts.append(f"{ik_failed} IK failed")
        if plan_failed:
            summary_parts.append(f"{plan_failed} plan/exec failed")

        hardware_issues = []
        current_mode = self._read_current_mode()
        if current_mode in ("real", "both"):
            for rn in ROBOT_CONFIG:
                if not self._robot_program_running.get(rn, False):
                    hardware_issues.append(rn)
        if hardware_issues:
            summary_parts.append(
                f"⚠ Hardware: {', '.join(hardware_issues)} disconnected"
            )

        summary = " | ".join(summary_parts)
        self.get_logger().info(
            f"Proto-sim DONE (legacy): {completed}/{total} ok, "
            f"{ik_failed} IK-fail, {plan_failed} plan-fail"
        )

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
            "stopped": self._proto_sim_stop,
            "hardware_issues": hardware_issues,
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
