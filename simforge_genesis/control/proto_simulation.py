"""Pose generation and execution helpers for the ``proto_sim`` workflow.

This module contains logic that is shared between the wx GUI implementation
and automated tests.  It is deliberately GUI-agnostic so that the behaviour
can be exercised programmatically.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core import CommandRejected, EventTopic
from ..core.commands import CartesianMoveCommand, JointTargetsCommand
from ..core.models import RobotProfile
from .session import SimulationSession


# ---------------------------------------------------------------------------
# Parameter / pose representations
# ---------------------------------------------------------------------------


ParameterMap = Dict[str, float]


@dataclass(frozen=True)
class ProtoSimParameters:
    """Fully expanded parameter sequences for pose sampling."""

    horiz: Sequence[float]
    vert: Sequence[float]
    distance: Sequence[float]
    roll: Sequence[float]
    pitch: Sequence[float]
    yaw: Sequence[float]


@dataclass(frozen=True)
class ProtoPose:
    """Pose expressed in the target object's reference frame."""

    parameters: ParameterMap
    position_m: Tuple[float, float, float]
    orientation_deg: Tuple[float, float, float]
    orientation_quat_wxyz: Tuple[float, float, float, float]


@dataclass
class ProtoPoseExecution:
    """Execution outcome for a single sampled pose."""

    pose: ProtoPose
    success: bool
    failure_reason: Optional[str] = None
    trajectory: Optional[Dict[str, Any]] = None
    duration_s: Optional[float] = None
    started_at_s: float = field(default_factory=time.time)
    finished_at_s: Optional[float] = None


@dataclass
class ObjectRunResult:
    """Aggregated execution data for a single object."""

    object_name: str
    reference_frame_key: str
    origin_in_base_m: Tuple[float, float, float]
    origin_orientation_rpy_deg: Tuple[float, float, float]
    poses: List[ProtoPoseExecution] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for pose in self.poses if pose.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for pose in self.poses if not pose.success)

    def to_plan_payload(self) -> Dict[str, Any]:
        """Convert successful poses into the exportable JSON structure."""
        pose_in_base = list(self.origin_in_base_m) + list(self.origin_orientation_rpy_deg)
        plan: Dict[str, Any] = {
            "pose_in_robot_frame": [float(v) for v in pose_in_base],
        }
        success_index = 1
        for execution in self.poses:
            if not execution.success or not execution.trajectory:
                continue
            key = f"pose_{success_index}"
            plan[key] = {
                "waypoints": execution.trajectory.get("waypoints", []),
                "protocol_pose": execution.pose.parameters,
            }
            success_index += 1
        return plan


@dataclass
class ProtoSimRunResult:
    """Full run result covering all selected objects."""

    robot_name: str
    objects: Dict[str, ObjectRunResult]
    home_trajectory: Optional[Dict[str, Any]] = None

    def total_successes(self) -> int:
        return sum(obj.success_count for obj in self.objects.values())

    def total_failures(self) -> int:
        return sum(obj.failure_count for obj in self.objects.values())

    def build_plan_export(self) -> Dict[str, Any]:
        """Return a dict ready to be dumped as the plan JSON file."""
        plan: Dict[str, Any] = {}
        for idx, (name, result) in enumerate(self.objects.items()):
            payload = result.to_plan_payload()
            if self.home_trajectory and idx == 0:
                payload["pose_home"] = {
                    "waypoints": self.home_trajectory.get("waypoints", []),
                    "protocol_pose": {},
                }
            plan[name] = payload
        return plan

    def iter_failures(self) -> Iterable[Tuple[str, int, ProtoPoseExecution]]:
        for name, result in self.objects.items():
            for index, execution in enumerate(result.poses, start=1):
                if not execution.success:
                    yield name, index, execution


@dataclass(frozen=True)
class ProtoSimProgress:
    """Progress callback payload emitted after each pose is handled."""

    object_name: str
    pose_index: int
    total_poses: int
    execution: ProtoPoseExecution


ProgressCallback = Callable[[ProtoSimProgress], None]



# ---------------------------------------------------------------------------
# Geometry helpers derived from the SpatialScout reference implementation
# ---------------------------------------------------------------------------


def _quat_from_matrix(R: List[List[float]]) -> Tuple[float, float, float, float]:
    """Compute quaternion (x, y, z, w) from a rotation matrix."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    else:
        diag = [m00, m11, m22]
        idx = max(range(3), key=lambda i: diag[i])
        if idx == 0:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            x = 0.25 * s
            w = (m21 - m12) / s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif idx == 1:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            y = 0.25 * s
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            z = 0.25 * s
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
    return (float(x), float(y), float(z), float(w))


def _normalize_quaternion_wxyz(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, x / norm, y / norm, z / norm)


def _xyzw_to_wxyz(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = quat
    return (w, x, y, z)


def _quat_to_rpy_deg(quat_wxyz: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    w, x, y, z = (float(v) for v in quat_wxyz)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _quaternion_to_matrix_wxyz(quat: Tuple[float, float, float, float]) -> List[List[float]]:
    w, x, y, z = quat
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _matrix_transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]


def _matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    result: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
    return result


def _mm_to_m(value: float) -> float:
    return value / 1000.0


def _rot_vec(pitch_deg: float, yaw_deg: float, dist_mm: float) -> Tuple[float, float, float]:
    d = _mm_to_m(dist_mm)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (
        sy * d * cp,
        cy * d * cp,
        d * sp,
    )


def _q_mul(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rpy_to_quat_xyzw(roll_rad: float, pitch_rad: float, yaw_rad: float) -> Tuple[float, float, float, float]:
    cr, sr = math.cos(roll_rad * 0.5), math.sin(roll_rad * 0.5)
    cp, sp = math.cos(pitch_rad * 0.5), math.sin(pitch_rad * 0.5)
    cy, sy = math.cos(yaw_rad * 0.5), math.sin(yaw_rad * 0.5)
    return (
        sr * cp * cy + cr * sp * sy,
        cr * sp * cy - sr * cp * sy,
        cr * cp * sy + sr * sp * cy,
        cr * cp * cy - sr * sp * sy,
    )


def _look_at(position: Tuple[float, float, float], pivot: Tuple[float, float, float], roll_deg: float) -> Tuple[float, float, float, float]:
    src = np.asarray(position, dtype=np.float64)
    tgt = np.asarray(pivot, dtype=np.float64)
    z_axis = tgt - src
    norm = np.linalg.norm(z_axis)
    if norm < 1e-9:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_axis = z_axis / norm

    up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(up_axis, z_axis)) >= 0.95:
        up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    quat_xyzw = _quat_from_matrix(rot_matrix.tolist())
    if abs(roll_deg) > 1e-6:
        roll_quat = _rpy_to_quat_xyzw(0.0, 0.0, math.radians(roll_deg))
        quat_xyzw = _q_mul(quat_xyzw, roll_quat)
    return quat_xyzw


def generate_proto_poses(params: ProtoSimParameters) -> List[ProtoPose]:
    """Generate pose samples mirroring the SpatialScout sampling pipeline."""

    poses: List[ProtoPose] = []
    for horiz in params.horiz:
        for vert in params.vert:
            pivot = (
                _mm_to_m(horiz),
                0.0,
                _mm_to_m(vert),
            )
            for pitch in params.pitch:
                for yaw in params.yaw:
                    for distance in params.distance:
                        for roll in params.roll:
                            offset = _rot_vec(pitch, yaw, distance)
                            position = (
                                pivot[0] + offset[0],
                                pivot[1] + offset[1],
                                pivot[2] + offset[2],
                            )
                            quat_xyzw = _look_at(position, pivot, roll)
                            quat_wxyz = _normalize_quaternion_wxyz(_xyzw_to_wxyz(quat_xyzw))
                            orientation = _quat_to_rpy_deg(quat_wxyz)
                            pose = ProtoPose(
                                parameters={
                                    "horiz": float(horiz),
                                    "vert": float(vert),
                                    "distance": float(distance),
                                    "roll": float(roll),
                                    "pitch": float(pitch),
                                    "yaw": float(yaw),
                                },
                                position_m=(float(position[0]), float(position[1]), float(position[2])),
                                orientation_deg=(float(orientation[0]), float(orientation[1]), float(orientation[2])),
                                orientation_quat_wxyz=(
                                    float(quat_wxyz[0]),
                                    float(quat_wxyz[1]),
                                    float(quat_wxyz[2]),
                                    float(quat_wxyz[3]),
                                ),
                            )
                            poses.append(pose)
    return poses


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _object_pose_in_base(
    robot: RobotProfile,
    frame_pose: Tuple[Sequence[float], Sequence[float]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    frame_pos, frame_quat = frame_pose
    base_pos = tuple(float(v) for v in robot.mount.position)
    base_orientation = tuple(float(v) for v in robot.mount.orientation)
    base_rot = _quaternion_to_matrix_wxyz(base_orientation)
    base_rot_T = _matrix_transpose(base_rot)

    relative = (
        float(frame_pos[0]) - base_pos[0],
        float(frame_pos[1]) - base_pos[1],
        float(frame_pos[2]) - base_pos[2],
    )
    x = base_rot_T[0][0] * relative[0] + base_rot_T[0][1] * relative[1] + base_rot_T[0][2] * relative[2]
    y = base_rot_T[1][0] * relative[0] + base_rot_T[1][1] * relative[1] + base_rot_T[1][2] * relative[2]
    z = base_rot_T[2][0] * relative[0] + base_rot_T[2][1] * relative[1] + base_rot_T[2][2] * relative[2]

    frame_rot = _quaternion_to_matrix_wxyz(tuple(float(v) for v in frame_quat))
    relative_rot = _matrix_multiply(base_rot_T, frame_rot)
    quat_xyzw = _quat_from_matrix(relative_rot)
    quat_wxyz = _normalize_quaternion_wxyz(_xyzw_to_wxyz(quat_xyzw))
    orientation_deg = _quat_to_rpy_deg(quat_wxyz)

    return (float(x), float(y), float(z)), orientation_deg


async def _register_subscription(bus, topic: str, handler: Callable[[Any], Awaitable[None] | None]) -> None:
    maybe_task = bus.subscribe(topic, handler)
    if asyncio.isfuture(maybe_task) or isinstance(maybe_task, asyncio.Task):
        await maybe_task


async def _unregister_subscription(bus, topic: str, handler: Callable[[Any], Awaitable[None] | None]) -> None:
    maybe_task = bus.unsubscribe(topic, handler)
    if asyncio.isfuture(maybe_task) or isinstance(maybe_task, asyncio.Task):
        await maybe_task


async def _execute_pose(
    session: SimulationSession,
    robot_name: str,
    frame_key: str,
    pose: ProtoPose,
    *,
    idle_timeout: float,
    logger,
    home_joints_deg: Optional[Sequence[float]] = None,
    enable_home_recovery: bool = True,
) -> ProtoPoseExecution:
    """Execute a single pose, with optional home recovery on failure.
    
    If a pose fails due to IK collision or planning failure and home_recovery is enabled,
    the robot will first go to the home position and retry the pose from there.
    The recorded trajectory will contain the full sequence: current→home→target.
    """
    execution = await _execute_pose_single_attempt(
        session, robot_name, frame_key, pose,
        idle_timeout=idle_timeout, logger=logger
    )
    
    # If successful or home recovery disabled, return as-is
    if execution.success or not enable_home_recovery or not home_joints_deg:
        return execution
    
    # Check if failure is recoverable (IK collision or planning failure)
    reason = execution.failure_reason or ""
    recoverable_failures = (
        "ik_goal_collision", "ik_failure", "ik_out_of_tol",
        "planner_collision", "planning_invalid", "planner_invalid",
        "planning_failure", "planner_failure"
    )
    if not any(r in reason.lower() for r in recoverable_failures):
        logger.debug(
            "[%s] Failure reason '%s' not recoverable via home; skipping retry",
            robot_name, reason
        )
        return execution
    
    logger.info(
        "[%s] Pose failed (%s); attempting home recovery for frame=%s",
        robot_name, reason, frame_key
    )
    
    # Step 1: Go to home position
    home_trajectory = await _execute_home_move(
        session, robot_name, home_joints_deg,
        idle_timeout=idle_timeout, logger=logger
    )
    
    if home_trajectory is None:
        logger.warning(
            "[%s] Home recovery failed - could not reach home position",
            robot_name
        )
        return execution  # Return original failure
    
    # Step 2: Retry the pose from home
    logger.info(
        "[%s] Reached home; retrying pose for frame=%s",
        robot_name, frame_key
    )
    retry_execution = await _execute_pose_single_attempt(
        session, robot_name, frame_key, pose,
        idle_timeout=idle_timeout, logger=logger
    )
    
    if not retry_execution.success:
        logger.info(
            "[%s] Home recovery failed - pose still unreachable after going home (reason=%s)",
            robot_name, retry_execution.failure_reason
        )
        # Return retry failure but note that home recovery was attempted
        retry_execution.failure_reason = f"{retry_execution.failure_reason}_after_home_retry"
        return retry_execution
    
    # Step 3: Combine trajectories (home trajectory + pose trajectory)
    logger.info(
        "[%s] Home recovery successful for frame=%s",
        robot_name, frame_key
    )
    combined_trajectory = _combine_trajectories(home_trajectory, retry_execution.trajectory)
    retry_execution.trajectory = combined_trajectory
    
    return retry_execution


def _combine_trajectories(
    first: Optional[Dict[str, Any]],
    second: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Combine two trajectory dicts into one with all waypoints in sequence."""
    if first is None:
        return second
    if second is None:
        return first
    
    first_waypoints = first.get("waypoints", [])
    second_waypoints = second.get("waypoints", [])
    
    combined_waypoints = list(first_waypoints) + list(second_waypoints)
    
    return {
        "waypoints": combined_waypoints,
        "home_recovery": True,  # Flag to indicate this trajectory involved home recovery
    }


async def _execute_pose_single_attempt(
    session: SimulationSession,
    robot_name: str,
    frame_key: str,
    pose: ProtoPose,
    *,
    idle_timeout: float,
    logger,
) -> ProtoPoseExecution:
    """Execute a single pose attempt without recovery logic."""
    execution = ProtoPoseExecution(pose=pose, success=False)
    rejection_event = asyncio.Event()
    rejection_reason: Dict[str, str] = {}

    async def _on_command(event):  # noqa: ANN001 - signature defined by bus
        if not isinstance(event, CommandRejected):
            return
        if event.robot != robot_name:
            return
        if getattr(event, "command_name", "") != "CartesianMoveCommand":
            return
        rejection_reason["reason"] = getattr(event, "reason", "unknown")
        if not rejection_reason["reason"] and event.payload:
            rejection_reason["reason"] = str(event.payload.get("reason", "unknown"))
        rejection_event.set()

    await session.clear_recent_trajectories(robot_name)
    await _register_subscription(session.event_bus, EventTopic.COMMAND.value, _on_command)
    try:
        command = CartesianMoveCommand(
            robot_name=robot_name,
            position_m=list(pose.position_m),
            orientation_deg=list(pose.orientation_deg),
            duration=4.0,
            reference_frame=frame_key,
            metadata={
                "source": "proto_sim",
                "object_frame": frame_key,
                "target_quat_wxyz": list(pose.orientation_quat_wxyz),
            },
        )
        await session.send_command(command)
        try:
            await session.wait_until_idle(robot_name, timeout=idle_timeout)
        except TimeoutError:
            execution.failure_reason = "timeout"
            logger.warning(
                "[%s] proto_sim pose timed out (frame=%s params=%s)",
                robot_name,
                frame_key,
                pose.parameters,
            )
            return execution

        if rejection_event.is_set():
            execution.failure_reason = rejection_reason.get("reason", "unknown")
            logger.info(
                "[%s] proto_sim pose rejected (frame=%s reason=%s params=%s)",
                robot_name,
                frame_key,
                execution.failure_reason,
                pose.parameters,
            )
            return execution

        trajectory = await session.pop_recent_trajectory(robot_name)
        execution.success = True
        execution.trajectory = trajectory
        execution.duration_s = float(command.duration)
        execution.finished_at_s = time.time()
        logger.debug(
            "[%s] proto_sim pose executed (frame=%s params=%s)",
            robot_name,
            frame_key,
            pose.parameters,
        )
        return execution
    finally:
        await _unregister_subscription(session.event_bus, EventTopic.COMMAND.value, _on_command)


async def _execute_home_move(
    session: SimulationSession,
    robot_name: str,
    joints_deg: Sequence[float],
    *,
    idle_timeout: float,
    logger,
) -> Optional[Dict[str, Any]]:
    if not joints_deg:
        logger.debug("[%s] No home joints specified; skipping home move", robot_name)
        return None

    rejection_event = asyncio.Event()
    rejection_reason: Dict[str, str] = {}

    async def _on_command(event):  # noqa: ANN001 - event signature
        if not isinstance(event, CommandRejected):
            return
        if event.robot != robot_name:
            return
        if getattr(event, "command_name", "") != "JointTargetsCommand":
            return
        rejection_reason["reason"] = getattr(event, "reason", "unknown")
        if not rejection_reason["reason"] and event.payload:
            rejection_reason["reason"] = str(event.payload.get("reason", "unknown"))
        rejection_event.set()

    await session.clear_recent_trajectories(robot_name)
    await _register_subscription(session.event_bus, EventTopic.COMMAND.value, _on_command)
    try:
        command = JointTargetsCommand(
            robot_name=robot_name,
            values_deg=[float(v) for v in joints_deg],
            duration=4.0,
            metadata={"source": "proto_sim_home"},
        )
        await session.send_command(command)
        try:
            await session.wait_until_idle(robot_name, timeout=idle_timeout)
        except TimeoutError:
            logger.warning("[%s] Home move timed out", robot_name)
            return None

        if rejection_event.is_set():
            logger.info(
                "[%s] Home move rejected: %s",
                robot_name,
                rejection_reason.get("reason", "unknown"),
            )
            return None

        trajectory = await session.pop_recent_trajectory(robot_name)
        logger.debug("[%s] Home move executed", robot_name)
        return trajectory
    finally:
        await _unregister_subscription(session.event_bus, EventTopic.COMMAND.value, _on_command)


async def execute_proto_sim(
    session: SimulationSession,
    robot: RobotProfile,
    frame_map: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    object_frames: Dict[str, str],
    poses: Sequence[ProtoPose],
    *,
    idle_timeout: float = 15.0,
    logger,
    progress: Optional[ProgressCallback] = None,
    enable_home_recovery: bool = True,
) -> ProtoSimRunResult:
    """Execute the sampled poses sequentially for each selected object.
    
    Args:
        enable_home_recovery: If True, when a pose fails due to IK collision or planning
            failure, the robot will first go to the home position and retry the pose.
            The trajectory will contain the full sequence (current→home→target).
    """

    # Prepare home joints for recovery
    home_joints_deg = list(robot.initial_joint_positions_deg or [])
    if not home_joints_deg:
        home_joints_deg = [0.0] * robot.joint_count
    if len(home_joints_deg) < robot.joint_count:
        home_joints_deg = home_joints_deg + [0.0] * (robot.joint_count - len(home_joints_deg))
    elif len(home_joints_deg) > robot.joint_count:
        home_joints_deg = home_joints_deg[: robot.joint_count]

    results: Dict[str, ObjectRunResult] = {}
    for object_name, frame_key in object_frames.items():
        frame_pose = frame_map.get(frame_key)
        if frame_pose is None:
            logger.warning("Reference frame '%s' not found; skipping %s", frame_key, object_name)
            continue

        origin_in_base, origin_rpy = _object_pose_in_base(robot, frame_pose)
        run_result = ObjectRunResult(
            object_name=object_name,
            reference_frame_key=frame_key,
            origin_in_base_m=origin_in_base,
            origin_orientation_rpy_deg=tuple(float(v) for v in origin_rpy),
        )
        results[object_name] = run_result

        total_poses = len(poses)
        for index, pose in enumerate(poses, start=1):
            execution = await _execute_pose(
                session,
                robot.name,
                frame_key,
                pose,
                idle_timeout=idle_timeout,
                logger=logger,
                home_joints_deg=home_joints_deg,
                enable_home_recovery=enable_home_recovery,
            )
            run_result.poses.append(execution)
            if progress is not None:
                try:
                    progress(ProtoSimProgress(object_name, index, total_poses, execution))
                except Exception:
                    logger.debug("proto_sim progress callback failed", exc_info=True)

    home_trajectory: Optional[Dict[str, Any]] = None
    try:
        home_trajectory = await _execute_home_move(
            session,
            robot.name,
            home_joints_deg,
            idle_timeout=idle_timeout,
            logger=logger,
        )
    except Exception:
        logger.exception("[%s] Home move failed", robot.name)

    return ProtoSimRunResult(robot_name=robot.name, objects=results, home_trajectory=home_trajectory)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def format_summary_lines(result: ProtoSimRunResult) -> List[str]:
    """Return a human-readable summary for log files."""
    lines = [
        "ProtoSim execution summary",
        f"Robot: {result.robot_name}",
        f"Total successes: {result.total_successes()}",
        f"Total failures: {result.total_failures()}",
        "",
        "Per-object results:",
    ]
    for name, obj in result.objects.items():
        lines.append(
            f"  - {name}: {obj.success_count} success / {obj.failure_count} failure (poses={len(obj.poses)})"
        )
    failures = list(result.iter_failures())
    if failures:
        lines.append("")
        lines.append("Failure details:")
        for object_name, index, execution in failures:
            lines.append(
                f"  * {object_name} pose {index}: reason={execution.failure_reason or 'unknown'} params={execution.pose.parameters}"
            )
    if result.home_trajectory:
        lines.append("")
        lines.append("Home trajectory recorded (pose_home entry available).")
    return lines


__all__ = [
    "ProtoSimParameters",
    "ProtoPose",
    "ProtoPoseExecution",
    "ObjectRunResult",
    "ProtoSimRunResult",
    "ProtoSimProgress",
    "ProgressCallback",
    "generate_proto_poses",
    "execute_proto_sim",
    "format_summary_lines",
]
