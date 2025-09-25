"""Robot coordinator orchestrating commands for a single robot."""
from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Dict, Tuple, Sequence

import math
import numpy as np

from ..core import (
    CartesianMoveCommand,
    JointTargetsCommand,
    StopCommand,
    Command,
    EventTopic,
    CommandRejected,
    CommandPriority,
)
from ..core.models import RobotProfile, MotionTarget, Pose
from ..core.config_schema import SafetyPolicy
from ..services import (
    IKSolver,
    IKRequest,
    MotionPlanner,
    PlanOutcome,
    PlanRequest,
    CollisionWorld,
    CollisionQuery,
)
from ..core import PlannerStrategy
from ..core.trajectories import JointTrajectory
from ..infrastructure import set_joint_positions
from .event_bus import EventBus
from .command_bus import CommandBus
from .synchronized_trajectory import SynchronizedTrajectoryManager
from ..infrastructure import get_joint_positions
from simforge.transformations import quaternion_to_rotation_matrix, quaternion_multiply, rpy_to_quaternion


def _normalize_quaternion(values: np.ndarray | Tuple[float, float, float, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return arr / norm


def _quaternion_conjugate(values: np.ndarray | Tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = map(float, values)
    return np.array([w, -x, -y, -z], dtype=np.float64)


@dataclass
class RobotContext:
    profile: RobotProfile
    entity: object
    joint_count: int
    logger: logging.Logger
    safety_policy: Optional[SafetyPolicy] = None


@dataclass
class _PoseValidationState:
    target: MotionTarget
    goal_joints: np.ndarray = field(repr=False)


@dataclass
class _ActiveMotionState:
    command_id: str
    command_name: str
    metadata: Dict[str, Any] = field(repr=False)
    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    cancelled: bool = False
    cancel_reason: Optional[str] = None


class CommandExecutionError(RuntimeError):
    """Raised when a command fails due to expected runtime conditions."""

    def __init__(self, message: str, *, reason: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason = reason or message


class RobotCoordinator:
    """Processes commands for one robot using IK, planning, and trajectory execution."""

    def __init__(
        self,
        ctx: RobotContext,
        *,
        ik_solver: IKSolver,
        planner: MotionPlanner,
        sync_trajectory_manager: SynchronizedTrajectoryManager,
        command_bus: CommandBus,
        event_bus: EventBus,
        collision_world: Optional[CollisionWorld] = None,
        state_provider: Optional[Callable[[str], Dict[str, Tuple[float, ...]]]] = None,
        frame_transforms: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.ctx = ctx
        self.ik_solver = ik_solver
        self.planner = planner
        self.sync_trajectory_manager = sync_trajectory_manager
        self.command_bus = command_bus
        self.event_bus = event_bus
        self.collision_world = collision_world
        self._state_provider = state_provider
        self.logger = logger or logging.getLogger(f"simforge.coordinator.{ctx.profile.name}")
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._frames = frame_transforms or {}
        if "world" not in self._frames:
            self._frames["world"] = (
                np.zeros(3, dtype=np.float64),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            )
        metadata = ctx.profile.metadata or {}
        self._ik_pos_tol = self._metadata_float(metadata, "legacy_control_ik_pos_tolerance_m", default=1e-3)
        self._ik_rot_tol = self._metadata_float(metadata, "legacy_control_ik_rot_tolerance_deg", default=1.0)
        self._ik_refine_pos_tol = self._metadata_float(
            metadata,
            "legacy_control_ik_refine_pos_tolerance_m",
            default=min(self._ik_pos_tol, 3e-4),
        )
        self._ik_refine_rot_tol = self._metadata_float(
            metadata,
            "legacy_control_ik_refine_rot_tolerance_deg",
            default=min(self._ik_rot_tol, 0.3),
        )
        self._ik_refine_max_attempts = max(0, int(float(metadata.get("legacy_control_ik_refine_max_attempts", 2))))
        self._pending_pose_validation: Optional[_PoseValidationState] = None
        self._pose_validation_task: Optional[asyncio.Task] = None
        self._last_validation_failure: Optional[Dict[str, object]] = None
        self._active_motion: Optional[_ActiveMotionState] = None

    def _queue_mode(self, metadata: Optional[Dict[str, Any]]) -> str:
        if not metadata:
            return "append"
        mode = metadata.get("queue_mode")
        if isinstance(mode, str) and mode:
            return mode.lower()
        if metadata.get("interrupt_active") or metadata.get("interrupt"):
            return "interrupt"
        return "append"

    async def _await_active_motion_if_needed(self, next_command: Command) -> None:
        motion = self._active_motion
        if motion is None:
            return

        if isinstance(next_command, StopCommand):
            self.logger.info(
                "[%s] Stop command received; cancelling active motion %s",
                self.ctx.profile.name,
                motion.command_name,
            )
            self._cancel_active_motion("stop_command")
            return

        queue_mode = self._queue_mode(next_command.metadata)
        if queue_mode == "interrupt" or next_command.priority >= CommandPriority.CRITICAL:
            self.logger.info(
                "[%s] Interrupting %s with %s",
                self.ctx.profile.name,
                motion.command_name,
                type(next_command).__name__,
            )
            self._cancel_active_motion("interrupt")
            return

        await motion.done.wait()

    def _start_active_motion(self, command: Command) -> _ActiveMotionState:
        if self._active_motion is not None:
            self.logger.debug(
                "[%s] Overwriting active motion %s with %s",
                self.ctx.profile.name,
                self._active_motion.command_name,
                type(command).__name__,
            )
            self._cancel_active_motion("overwrite")
        state = _ActiveMotionState(
            command_id=command.command_id,
            command_name=type(command).__name__,
            metadata=dict(command.metadata or {}),
        )
        self._active_motion = state
        return state

    def _cancel_active_motion(self, reason: str) -> None:
        motion = self._active_motion
        if motion is None:
            return
        motion.cancelled = True
        motion.cancel_reason = reason
        if not motion.done.is_set():
            motion.done.set()
        self._active_motion = None
        self._pending_pose_validation = None
        try:
            current = self._current_joints(self.ctx.joint_count)
            if current.size:
                self.sync_trajectory_manager.hold_position(self.ctx.profile.name, current)
                try:
                    set_joint_positions(
                        self.ctx.entity,
                        current.tolist(),
                        self.ctx.joint_count,
                        self.ctx.logger,
                    )
                except Exception:  # pragma: no cover - defensive
                    self.logger.debug(
                        "[%s] Failed to apply hold joints during cancel", self.ctx.profile.name,
                        exc_info=True,
                    )
        except Exception:  # pragma: no cover - defensive
            self.logger.debug(
                "[%s] Failed to snapshot current joints during cancel", self.ctx.profile.name,
                exc_info=True,
            )
            try:
                self.sync_trajectory_manager.stop_trajectory(self.ctx.profile.name)
            except Exception:  # pragma: no cover - defensive
                self.logger.debug(
                    "[%s] Failed to stop active trajectory during cancel", self.ctx.profile.name,
                    exc_info=True,
                )

    def has_active_motion(self) -> bool:
        motion = self._active_motion
        return bool(motion is not None and not motion.done.is_set())

    def start(self) -> None:
        if self._task and not self._task.done():
            self.logger.warning(f"Coordinator for {self.ctx.profile.name} already running")
            return
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())
        self.logger.info(f"Started coordinator for {self.ctx.profile.name}")

    async def stop(self) -> None:
        self._stop.set()
        self._cancel_active_motion("coordinator_stop")
        if not self.command_bus.is_stopped():
            try:
                await self.command_bus.submit(
                    StopCommand(
                        robot_name=self.ctx.profile.name,
                        priority=CommandPriority.EMERGENCY_STOP,
                        metadata={"reason": "coordinator_stop"},
                    )
                )
            except Exception:
                self.logger.debug("Failed to enqueue stop command for %s", self.ctx.profile.name)
        if self._task:
            await self._task
        if self._pose_validation_task and not self._pose_validation_task.done():
            self._pose_validation_task.cancel()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            command = await self.command_bus.get(self.ctx.profile.name)
            await self._await_active_motion_if_needed(command)
            try:
                if isinstance(command, StopCommand):
                    break
                await self._process_command(command)
            except CommandExecutionError as exc:
                await self.event_bus.publish(
                    CommandRejected(
                        topic=EventTopic.COMMAND,
                        robot=command.robot,
                        command_name=type(command).__name__,
                        reason=exc.reason,
                    )
                )
                self.logger.warning(str(exc))
            except Exception as exc:
                await self.event_bus.publish(
                    CommandRejected(
                        topic=EventTopic.COMMAND,
                        robot=command.robot,
                        command_name=type(command).__name__,
                        reason=str(exc),
                    )
                )
                self.logger.exception("Failed to process command %s", command)

    async def _process_command(self, command: Command) -> None:
        if isinstance(command, JointTargetsCommand):
            await self._handle_joint_targets(command)
        elif isinstance(command, CartesianMoveCommand):
            await self._handle_cartesian(command)
        else:
            raise ValueError(f"Unsupported command {command}")

    async def _handle_joint_targets(self, command: JointTargetsCommand) -> None:
        self._pending_pose_validation = None
        self._last_validation_failure = None
        robot = self.ctx.profile
        values_deg = tuple(float(v) for v in command.values_deg)
        target_rad = np.deg2rad(values_deg)
        start_rad = self._current_joints(len(target_rad))

        if target_rad.size == 0:
            self.logger.debug("Joint target command with no values; skipping")
            return

        if np.allclose(start_rad[: target_rad.size], target_rad, atol=1e-6):
            self.logger.debug("Joint targets identical to current state; skipping move")
            return

        metadata = command.metadata or {}
        if metadata.get("direct_joint_set"):
            self.logger.debug(
                "Direct joint set for %s from GUI: goal=%s",
                robot.name,
                np.round(values_deg, 3),
            )
            self.sync_trajectory_manager.hold_position(robot.name, target_rad)
            set_joint_positions(
                self.ctx.entity,
                target_rad.tolist(),
                self.ctx.joint_count,
                self.ctx.logger,
            )
            return

        duration = max(float(command.duration or 0.5), 0.05)
        joint_names = tuple(f"j{idx}" for idx in range(target_rad.size))
        trajectory = JointTrajectory(
            joint_names=joint_names,
            times_s=(0.0, duration),
            positions=(tuple(float(v) for v in start_rad[: target_rad.size]), tuple(float(v) for v in target_rad)),
        )

        self.logger.debug(
            "Executing direct joint move for %s: start=%s → goal=%s (duration=%.2fs)",
            robot.name,
            np.round(np.rad2deg(start_rad[: target_rad.size]), 3),
            np.round(values_deg, 3),
            duration,
        )

        self._start_active_motion(command)
        self.sync_trajectory_manager.start_trajectory(robot.name, trajectory)

    async def _handle_cartesian(self, command: CartesianMoveCommand) -> None:
        self._start_active_motion(command)
        self._pending_pose_validation = None
        self._last_validation_failure = None
        robot = self.ctx.profile
        target_pose = self._motion_target_from_cartesian(command)
        self._enforce_workspace(target_pose.pose.position, frame=target_pose.frame)
        validator = self._state_validator()
        seed = tuple(float(v) for v in self._current_joints(robot.joint_count))
        target = IKRequest(robot=robot, target=target_pose, seed=seed, is_state_valid=validator)
        self.logger.debug(
            "IK target for %s: pos=%s, orient=%s",
            robot.name,
            target_pose.pose.position,
            target_pose.pose.orientation,
        )
        try:
            ik_result = await self._run_blocking(self.ik_solver.solve, target)
        except Exception:
            self._cancel_active_motion("ik_exception")
            raise
        if not ik_result.success or ik_result.solution is None:
            self._cancel_active_motion("ik_failure")
            message = self._format_ik_failure(command, target_pose, ik_result)
            raise CommandExecutionError(message, reason=self._ik_failure_reason(ik_result))

        solution = np.asarray(ik_result.solution, dtype=np.float64)
        best_pos_err, best_ang_err = self._compute_pose_error(solution, target_pose.pose)
        if best_pos_err > self._ik_pos_tol + 1e-6 or best_ang_err > self._ik_rot_tol + 1e-6:
            self.logger.debug(
                "%s IK solution outside tolerance: pos=%.4f m, ang=%.3f° (limits %.4f m / %.3f°)",
                robot.name,
                best_pos_err,
                best_ang_err,
                self._ik_pos_tol,
                self._ik_rot_tol,
            )
            refinement_state = _PoseValidationState(target=target_pose, goal_joints=solution.copy())
            attempts = 0
            while attempts < self._ik_refine_max_attempts:
                attempts += 1
                refined = await self._attempt_pose_refine(solution, refinement_state)
                if refined is None:
                    break
                solution, best_pos_err, best_ang_err = refined
                refinement_state.goal_joints = solution
                if best_pos_err <= self._ik_pos_tol + 1e-6 and best_ang_err <= self._ik_rot_tol + 1e-6:
                    self.logger.info(
                        "%s IK solution refined before planning: pos=%.4f m, ang=%.3f°",
                        robot.name,
                        best_pos_err,
                        best_ang_err,
                    )
                    break
            if best_pos_err > self._ik_pos_tol + 1e-6 or best_ang_err > self._ik_rot_tol + 1e-6:
                self._cancel_active_motion("ik_out_of_tol")
                message = self._format_ik_failure(
                    command,
                    target_pose,
                    ik_result,
                    pos_err=best_pos_err,
                    ang_err=best_ang_err,
                )
                raise CommandExecutionError(message, reason=self._ik_failure_reason(ik_result))

        goal_solution = tuple(float(v) for v in solution)
        start = tuple(self._current_joints(len(goal_solution)))
        self.logger.debug(
            "Path planning for %s: start=%s, goal=%s",
            robot.name,
            np.round(start, 3),
            np.round(solution, 3),
        )
        plan_req = PlanRequest(
            robot=robot,
            start=start,
            goal=goal_solution,
            strategy=PlannerStrategy.CARTESIAN_PREFERRED,
            timeout_s=3.0,
            is_state_valid=validator,
        )
        try:
            plan = await self._run_blocking(self.planner.plan, plan_req)
        except Exception:
            self._cancel_active_motion("plan_exception")
            raise
        if plan.outcome != PlanOutcome.SUCCESS or plan.trajectory is None:
            self._cancel_active_motion(f"planner_{plan.outcome.value}" if plan.outcome else "planner_failure")
            message = self._format_plan_failure(command, target_pose, plan_req, plan)
            self._pending_pose_validation = None
            raise CommandExecutionError(
                message,
                reason=f"planner_{plan.outcome.value}" if plan.outcome else "planning_failure",
            )
        # Use synchronized trajectory execution instead of async
        self.logger.info(f"Starting synchronized trajectory for {robot.name} (Cartesian): duration={plan.trajectory.duration:.3f}s")
        plan_goal = np.asarray(plan.trajectory.positions[-1], dtype=np.float64)
        if plan_goal.shape == solution.shape and np.linalg.norm(plan_goal - solution) > 5e-4:
            self.logger.debug(
                "%s planner terminal state differs from IK goal: |Δq|=%.4f rad",
                robot.name,
                float(np.linalg.norm(plan_goal - solution)),
            )
        self._pending_pose_validation = _PoseValidationState(
            target=target_pose,
            goal_joints=solution.copy(),
        )
        self.sync_trajectory_manager.start_trajectory(robot.name, plan.trajectory)
        self._last_validation_failure = None

    def _current_joints(self, dof: int) -> np.ndarray:
        arr = get_joint_positions(self.ctx.entity, self.ctx.logger, prefer_struct=True)
        
        # Check if we got valid (finite) joint positions
        if arr.size >= dof and np.all(np.isfinite(arr)):
            return arr[:dof]
        
        if arr.size > 0:
            # Check if any values are finite
            valid_mask = np.isfinite(arr)
            if np.any(valid_mask):
                padded = np.zeros(dof, dtype=np.float64)
                valid_count = min(arr.size, dof)
                for i in range(valid_count):
                    if np.isfinite(arr[i]):
                        padded[i] = arr[i]
                    elif self.ctx.profile.initial_joint_positions_deg and i < len(self.ctx.profile.initial_joint_positions_deg):
                        padded[i] = np.deg2rad(self.ctx.profile.initial_joint_positions_deg[i])
                return padded
        
        # Fall back to robot initial joints if Genesis values are invalid
        if self.ctx.profile.initial_joint_positions_deg:
            initial_rad = np.deg2rad(np.asarray(self.ctx.profile.initial_joint_positions_deg, dtype=np.float64))
            if initial_rad.size >= dof:
                return initial_rad[:dof]
            # Pad if needed
            padded = np.zeros(dof, dtype=np.float64)
            padded[:min(initial_rad.size, dof)] = initial_rad[:min(initial_rad.size, dof)]
            return padded
        
        self.logger.warning(f"No valid joint positions found for {self.ctx.profile.name}, using zeros")
        return np.zeros(dof, dtype=np.float64)

    def _motion_target_from_cartesian(self, command: CartesianMoveCommand) -> MotionTarget:
        frame_key = (command.reference_frame or "base").strip().lower()
        base_pos = np.asarray(self.ctx.profile.mount.position, dtype=np.float64)
        base_quat = _normalize_quaternion(self.ctx.profile.mount.orientation)
        base_rot = quaternion_to_rotation_matrix(base_quat)
        base_rot_T = base_rot.T

        local_pos = np.asarray([float(v) for v in command.position_m], dtype=np.float64)
        roll, pitch, yaw = (math.radians(float(v)) for v in command.orientation_deg)
        local_quat = _normalize_quaternion(rpy_to_quaternion(roll, pitch, yaw))

        base_aliases = {"", "base", "robot", "robot_base"}
        resolved_frame = "base"

        if frame_key in base_aliases:
            world_pos = base_rot.dot(local_pos) + base_pos
            world_quat = quaternion_multiply(base_quat, local_quat)
        else:
            frame = self._frames.get(frame_key)
            if frame is None:
                self.logger.warning(
                    "[%s] Unknown reference frame '%s'; defaulting to world frame",
                    self.ctx.profile.name,
                    command.reference_frame,
                )
                frame = self._frames.get("world")
                resolved_frame = "world"
            else:
                resolved_frame = frame_key
            frame_pos, frame_quat = frame
            frame_pos = np.asarray(frame_pos, dtype=np.float64)
            frame_quat = _normalize_quaternion(frame_quat)
            frame_rot = quaternion_to_rotation_matrix(frame_quat)
            world_pos = frame_rot.dot(local_pos) + frame_pos
            world_quat = quaternion_multiply(frame_quat, local_quat)

        world_quat = _normalize_quaternion(world_quat)

        pos_base = base_rot_T.dot(world_pos - base_pos)
        quat_base = quaternion_multiply(_quaternion_conjugate(base_quat), world_quat)
        quat_base = _normalize_quaternion(quat_base)

        self.logger.info(
            "[%s] Target in base frame (frame=%s): pos=%s m, quat_wxyz=%s",
            self.ctx.profile.name,
            resolved_frame,
            np.round(pos_base, 4),
            np.round(quat_base, 4),
        )

        pose = Pose(position=tuple(float(v) for v in pos_base), orientation=tuple(float(v) for v in quat_base))
        return MotionTarget(pose=pose, frame=resolved_frame)

    def _state_validator(self) -> Callable[[tuple[float, ...]], bool]:
        if self.collision_world is None:
            return lambda q: True

        # Get current robot joint state for comparison
        current_joints = self._current_joints(6)  # Most robots have 6 DOF
        current_state = tuple(float(v) for v in current_joints)

        def _check(q: tuple[float, ...]) -> bool:
            # Be more permissive for states close to current position (legacy behavior)
            if len(q) == len(current_state):
                diff = sum(abs(a - b) for a, b in zip(q, current_state))
                if diff < 0.1:  # Within ~5 degrees total difference
                    self._last_validation_failure = None
                    return True  # Always allow states close to current
            
            other_states: Dict[str, Tuple[float, ...]] = {}
            if self._state_provider is not None:
                try:
                    other_states = self._state_provider(self.ctx.profile.name)
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.debug("State provider failed: %s", exc)
            query = CollisionQuery(
                robot=self.ctx.profile,
                joints=q,
                other_robot_states=other_states,
            )
            try:
                result = self.collision_world.is_state_valid(query)
            except Exception as exc:  # pragma: no cover - defensive path
                self.logger.debug("Collision validation failed: %s", exc)
                self._last_validation_failure = None
                return True  # Be permissive on validation errors (legacy behavior)
            if result.in_collision:
                self._last_validation_failure = {
                    "kind": "collision",
                    "details": dict(result.details or {}),
                    "joints": tuple(float(v) for v in q),
                }
            else:
                self._last_validation_failure = None
            return not result.in_collision

        return _check

    def _enforce_workspace(self, position: Tuple[float, float, float], *, frame: str) -> None:
        safety = self.ctx.safety_policy
        if safety is None or not safety.enforce_workspace_bounds:
            return
        frame_name = frame or "base"
        min_xyz = safety.workspace_min
        max_xyz = safety.workspace_max
        axis_names = ("x", "y", "z")
        tolerance = 1e-6
        for axis, value, lo, hi in zip(axis_names, position, min_xyz, max_xyz):
            if value < lo - tolerance:
                message = (
                    f"Workspace violation: {axis}={value:.3f} m below minimum {lo:.3f} m (frame={frame_name})"
                )
                raise CommandExecutionError(message, reason=f"workspace_{axis}_below_min")
            if value > hi + tolerance:
                message = (
                    f"Workspace violation: {axis}={value:.3f} m above maximum {hi:.3f} m (frame={frame_name})"
                )
                raise CommandExecutionError(message, reason=f"workspace_{axis}_above_max")
        if position[2] < safety.minimum_clearance_m - tolerance:
            message = (
                f"Workspace violation: z={position[2]:.3f} m below minimum clearance "
                f"{safety.minimum_clearance_m:.3f} m (frame={frame_name})"
            )
            raise CommandExecutionError(message, reason="workspace_clearance_violation")

    def on_trajectory_completed(self, final_sample: np.ndarray) -> None:
        motion = self._active_motion
        if motion is not None:
            if not motion.done.is_set():
                motion.done.set()
            cancelled = motion.cancelled
            self._active_motion = None
        else:
            cancelled = False

        if cancelled:
            self.logger.debug(
                "[%s] Active motion %s cancelled (%s)",
                self.ctx.profile.name,
                motion.command_name if motion else "<unknown>",
                motion.cancel_reason if motion else "cancelled",
            )
            self._pending_pose_validation = None
            return

        if self._pending_pose_validation is None or self._pending_pose_validation.target.pose is None:
            return
        if self._pose_validation_task and not self._pose_validation_task.done():
            self._pose_validation_task.cancel()
        loop = asyncio.get_event_loop()
        self._pose_validation_task = loop.create_task(self._validate_pose_async(final_sample))

    async def _validate_pose_async(self, final_sample: np.ndarray) -> None:
        state_ref = self._pending_pose_validation
        await asyncio.sleep(0.0)
        state = self._pending_pose_validation
        if state is None or state.target.pose is None or state is not state_ref:
            return
        pose = state.target.pose
        if pose is None:
            self._pending_pose_validation = None
            return

        actual = self._current_joints(len(final_sample))
        pos_err, ang_err = self._compute_pose_error(actual, pose)
        if pos_err <= self._ik_pos_tol + 1e-6 and ang_err <= self._ik_rot_tol + 1e-6:
            self.logger.debug(
                "%s pose within tolerance after trajectory: pos=%.4f m, ang=%.3f°",
                self.ctx.profile.name,
                pos_err,
                ang_err,
            )
            self._pending_pose_validation = None
            return

        seeds: list[np.ndarray] = []
        goal = state.goal_joints
        if goal is not None and goal.size:
            goal_pos_err, goal_ang_err = self._compute_pose_error(goal, pose)
            if goal_pos_err <= self._ik_pos_tol + 1e-6 and goal_ang_err <= self._ik_rot_tol + 1e-6:
                self.logger.info(
                    "%s applying planned goal joints to correct pose drift: pos=%.4f m, ang=%.3f°",
                    self.ctx.profile.name,
                    goal_pos_err,
                    goal_ang_err,
                )
                self.sync_trajectory_manager.hold_position(self.ctx.profile.name, goal)
                set_joint_positions(
                    self.ctx.entity,
                    goal.tolist(),
                    self.ctx.joint_count,
                    self.ctx.logger,
                )
                actual = self._current_joints(len(goal))
                pos_err, ang_err = self._compute_pose_error(actual, pose)
                if pos_err <= self._ik_pos_tol + 1e-6 and ang_err <= self._ik_rot_tol + 1e-6:
                    self.logger.info(
                        "%s pose corrected by planned goal: pos=%.4f m, ang=%.3f°",
                        self.ctx.profile.name,
                        pos_err,
                        ang_err,
                    )
                    self._pending_pose_validation = None
                    return
                self.logger.debug(
                    "%s pose still outside tolerance after applying planned goal: pos=%.4f m, ang=%.3f°",
                    self.ctx.profile.name,
                    pos_err,
                    ang_err,
                )
                seeds.append(goal)
            else:
                self.logger.debug(
                    "%s planned goal joints outside tolerance: pos=%.4f m, ang=%.3f°",
                    self.ctx.profile.name,
                    goal_pos_err,
                    goal_ang_err,
                )
                seeds.append(goal)

        self.logger.info(
            "%s pose outside tolerance (pos=%.4f m, ang=%.3f°); attempting refine",
            self.ctx.profile.name,
            pos_err,
            ang_err,
        )

        attempts = 0
        seeds.append(actual)
        if not seeds:
            seeds.append(actual)

        while attempts < self._ik_refine_max_attempts:
            attempts += 1
            seed = seeds[(attempts - 1) % len(seeds)]
            refined = await self._attempt_pose_refine(np.asarray(seed, dtype=np.float64), state)
            if refined is None:
                continue
            new_q, pos_err_ref, ang_err_ref = refined
            self.sync_trajectory_manager.hold_position(self.ctx.profile.name, new_q)
            set_joint_positions(
                self.ctx.entity,
                new_q.tolist(),
                self.ctx.joint_count,
                self.ctx.logger,
            )
            actual = self._current_joints(len(new_q))
            pos_err, ang_err = pos_err_ref, ang_err_ref
            if pos_err <= self._ik_pos_tol + 1e-6 and ang_err <= self._ik_rot_tol + 1e-6:
                self.logger.info(
                    "%s pose refine succeeded: pos=%.4f m (limit %.4f), ang=%.3f° (limit %.3f°)",
                    self.ctx.profile.name,
                    pos_err,
                    self._ik_pos_tol,
                    ang_err,
                    self._ik_rot_tol,
                )
                self._pending_pose_validation = None
                return

        self.logger.warning(
            "%s pose still outside tolerance after %d refine attempts: pos=%.4f m, ang=%.3f°",
            self.ctx.profile.name,
            attempts,
            pos_err,
            ang_err,
        )
        self._pending_pose_validation = None

    def _compute_pose_error(self, joints: np.ndarray, target_pose: Pose) -> Tuple[float, float]:
        try:
            pos_current, quat_current = self.ik_solver.forward_kinematics(joints)
        except Exception as exc:
            self.logger.debug("Forward kinematics failed for %s: %s", self.ctx.profile.name, exc)
            return float("inf"), float("inf")

        target_pos = np.asarray(target_pose.position, dtype=np.float64)
        target_quat = np.asarray(target_pose.orientation, dtype=np.float64)
        pos_err = float(np.linalg.norm(pos_current - target_pos))

        q_cur = np.asarray(quat_current, dtype=np.float64)
        if np.linalg.norm(q_cur) > 0:
            q_cur = q_cur / np.linalg.norm(q_cur)
        if np.linalg.norm(target_quat) > 0:
            target_quat = target_quat / np.linalg.norm(target_quat)
        dot = float(np.dot(q_cur, target_quat))
        if dot < 0.0:
            dot = -dot
        dot = float(max(-1.0, min(1.0, dot)))
        ang_err = float(2.0 * math.degrees(math.acos(dot)))
        return pos_err, ang_err

    async def _attempt_pose_refine(
        self,
        seed: np.ndarray,
        state: _PoseValidationState,
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        pose = state.target.pose
        if pose is None:
            return None
        request = IKRequest(
            robot=self.ctx.profile,
            target=state.target,
            seed=tuple(float(v) for v in seed),
            is_state_valid=self._state_validator(),
            prefer_cartesian=True,
            position_tolerance_m=self._ik_refine_pos_tol,
            orientation_tolerance_deg=self._ik_refine_rot_tol,
            timeout_s=1.0,
        )
        try:
            result = await self._run_blocking(self.ik_solver.solve, request)
        except Exception as exc:
            self.logger.warning("Refine IK failed for %s: %s", self.ctx.profile.name, exc)
            return None
        if not result.success or result.solution is None:
            self.logger.debug("Refine IK returned no solution for %s", self.ctx.profile.name)
            return None
        new_q = np.asarray(result.solution, dtype=np.float64)
        pos_err, ang_err = self._compute_pose_error(new_q, pose)
        return new_q, pos_err, ang_err

    def _format_joint_vector(self, joints: Sequence[float]) -> str:
        values = tuple(float(v) for v in joints)
        if not values:
            return "[]"
        degs = [math.degrees(v) for v in values]
        if len(degs) > 6:
            display = ", ".join(f"{val:.1f}°" for val in degs[:6]) + ", …"
        else:
            display = ", ".join(f"{val:.1f}°" for val in degs)
        return f"[{display}]"

    def _describe_collision(self, failure: Optional[Dict[str, object]]) -> Optional[str]:
        if not failure or failure.get("kind") != "collision":
            return None
        details_obj = failure.get("details") or {}
        details = details_obj if isinstance(details_obj, dict) else {}
        collision_obj = details.get("collision") or {}
        collision = collision_obj if isinstance(collision_obj, dict) else {}

        segments: list[str] = []
        collision_type = collision.get("type")
        if collision_type:
            segments.append(str(collision_type))

        robot_link = collision.get("robot_link")
        obstacle = collision.get("obstacle")
        obstacle_link = collision.get("obstacle_link")
        links = collision.get("links")
        if not robot_link and isinstance(links, (list, tuple)) and len(links) == 2:
            robot_link, obstacle = links
        if robot_link and obstacle:
            counterpart = f"{obstacle}/{obstacle_link}" if obstacle_link else str(obstacle)
            segments.append(f"{robot_link} vs {counterpart}")
        elif robot_link:
            segments.append(str(robot_link))
        elif obstacle:
            segments.append(str(obstacle))

        timestamp = collision.get("timestamp")
        if isinstance(timestamp, (int, float)):
            segments.append(f"t={timestamp:.2f}s")

        joints = failure.get("joints")
        if isinstance(joints, (list, tuple)) and joints:
            segments.append(f"joints={self._format_joint_vector(joints)}")

        source = details.get("source")
        prefix = f"{source} collision" if source else "collision"
        if not segments:
            return prefix
        return f"{prefix}: {'; '.join(segments)}"

    def _ik_failure_reason(self, result) -> str:
        raw = result.raw if isinstance(result.raw, dict) else {}
        if raw:
            reason = raw.get("reason")
            if isinstance(reason, str) and reason:
                return f"ik_{reason}"
        if self._last_validation_failure and self._last_validation_failure.get("kind") == "collision":
            return "ik_goal_collision"
        return "ik_failure"

    def _format_ik_failure(
        self,
        command: CartesianMoveCommand,
        target: MotionTarget,
        result,
        *,
        pos_err: Optional[float] = None,
        ang_err: Optional[float] = None,
    ) -> str:
        robot_name = self.ctx.profile.name
        reasons: list[str] = []

        collision_text = self._describe_collision(self._last_validation_failure)
        if not collision_text and isinstance(result.raw, dict):
            collision_text = self._describe_collision(result.raw.get("failure"))
        if collision_text:
            reasons.append(collision_text)

        metrics = result.metrics
        if metrics is not None:
            if math.isfinite(metrics.position_error_m) or math.isfinite(metrics.orientation_error_deg):
                reasons.append(
                    "solver_error=({:.4f} m, {:.3f}°)".format(
                        float(metrics.position_error_m),
                        float(metrics.orientation_error_deg),
                    )
                )
            if metrics.diagnostics:
                reasons.append(f"metrics_diag={metrics.diagnostics}")

        raw = result.raw if isinstance(result.raw, dict) else None
        if raw:
            reason = raw.get("reason")
            if reason:
                reasons.append(f"solver reason={reason}")
            status = raw.get("status")
            if status:
                reasons.append(f"status={status}")
            diagnostics = {k: v for k, v in raw.items() if k not in {"reason", "status"}}
            if diagnostics:
                reasons.append(f"diagnostics={diagnostics}")
        elif result.raw is not None:
            reasons.append(f"solver diagnostics={result.raw}")

        if pos_err is not None or ang_err is not None:
            reasons.append(
                "pose_error=({:.4f} m, {:.3f}°)".format(
                    float(pos_err) if pos_err is not None else float("nan"),
                    float(ang_err) if ang_err is not None else float("nan"),
                )
            )

        if not reasons:
            reasons.append("solver returned no valid joint solution")

        pose = target.pose
        if pose is not None:
            pos = np.round(np.asarray(pose.position, dtype=np.float64), 4).tolist()
            quat = np.round(np.asarray(pose.orientation, dtype=np.float64), 4).tolist()
            frame = target.frame or "base"
            reasons.append(f"target pose frame={frame} pos={pos} quat_wxyz={quat}")

        return f"[{robot_name}] Cartesian IK failed: {'; '.join(reasons)}"

    def _format_plan_failure(
        self,
        command: CartesianMoveCommand,
        target: MotionTarget,
        request: PlanRequest,
        plan,
    ) -> str:
        robot_name = self.ctx.profile.name
        reasons: list[str] = []

        collision_text = self._describe_collision(self._last_validation_failure)
        raw = plan.raw if isinstance(plan.raw, dict) else None
        raw_reason = raw.get("reason") if raw else None

        if plan.outcome in (PlanOutcome.COLLISION, PlanOutcome.INVALID_GOAL) or (
            isinstance(raw_reason, str) and "collision" in raw_reason
        ):
            if collision_text:
                reasons.append(collision_text)

        if raw_reason:
            reasons.append(f"planner reason={raw_reason}")
        if raw:
            diagnostics = {k: v for k, v in raw.items() if k != "reason"}
            if diagnostics:
                reasons.append(f"planner diagnostics={diagnostics}")

        if plan.metrics is not None:
            reasons.append(
                f"planner={plan.metrics.planner_name}, states_checked={plan.metrics.states_validated}"
            )

        if not reasons:
            reasons.append("planner returned no trajectory")

        reasons.append(
            f"start={self._format_joint_vector(request.start)} -> goal={self._format_joint_vector(request.goal)}"
        )

        pose = target.pose
        if pose is not None:
            pos = np.round(np.asarray(pose.position, dtype=np.float64), 4).tolist()
            quat = np.round(np.asarray(pose.orientation, dtype=np.float64), 4).tolist()
            frame = target.frame or "base"
            reasons.append(f"target pose frame={frame} pos={pos} quat_wxyz={quat}")

        return f"[{robot_name}] Motion plan failed ({plan.outcome.value}): {'; '.join(reasons)}"

    @staticmethod
    def _metadata_float(metadata: Dict[str, str], key: str, *, default: float) -> float:
        try:
            value = metadata.get(key)
            return float(value) if value is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    async def _run_blocking(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, call)


__all__ = ["RobotCoordinator", "RobotContext"]
