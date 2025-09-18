"""High-level movement controller orchestrating robots, IK, and planning."""
from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..config_reader import SimforgeConfig
from ..genesis_renderer import GenesisRenderer
from ..logging_utils import setup_logging
from ..ik_drake import DrakeIKCache, DrakeIKOptions, solve_ik_drake
from .commands import (
    Command,
    ControlMode,
    SetJointCommand,
    SetJointTargetsCommand,
    CartesianMoveCommand,
    SwitchModeCommand,
)
from .robot_runtime import RobotRuntime, ActiveTrajectory
from . import collision
from . import scene_builder
from .robot_io import get_robot_joints, set_robot_joints
from .utils import deg_to_rad_list, rad_to_deg_list
from .ik_planner import plan_cartesian_move
from ..transformations import quaternion_to_rotation_matrix


class _RuntimeMirror(dict):
    """Dictionary-like view that keeps controller runtimes in sync."""

    def __init__(self, controller: "MovementController", getter, setter=None):
        super().__init__()
        self._controller = controller
        self._getter = getter
        self._setter = setter

    def refresh(self) -> None:
        super().clear()
        for name, runtime in self._controller.robots.items():
            super().__setitem__(name, self._getter(runtime))

    def __setitem__(self, key, value):
        if self._setter is not None and key in self._controller.robots:
            self._setter(self._controller.robots[key], value)
            self._controller._sync_legacy_views()
        else:
            super().__setitem__(key, value)


class MovementController:
    """High-level movement controller managing robot motion."""

    def __init__(self, config: SimforgeConfig, debug: bool = False):
        self.config = config
        self.logger = setup_logging(debug)
        self.renderer = GenesisRenderer(config.scene.backend, self.logger)

        self.robots: Dict[str, RobotRuntime] = {}
        for robot_config in self.config.robots:
            runtime = RobotRuntime(
                name=robot_config.name,
                config=robot_config,
                joint_targets=list(robot_config.initial_joint_positions or [0.0] * 6),
            )
            self._initialize_runtime(runtime)
            self.robots[runtime.name] = runtime

        collision.register_env_robots(self.robots, self.logger)
        collision.log_collision_status(self.robots, self.logger)

        self.robot_modes = _RuntimeMirror(self, lambda rt: rt.mode, setter=lambda rt, v: setattr(rt, "mode", v))
        self.joint_targets = _RuntimeMirror(self, lambda rt: rt.joint_targets, setter=lambda rt, v: setattr(rt, "joint_targets", list(v)))
        self.robot_entities = _RuntimeMirror(self, lambda rt: rt.entity)
        self.collision_checkers = _RuntimeMirror(self, lambda rt: rt.collision_checker)
        self.drake_caches = _RuntimeMirror(self, lambda rt: rt.drake_cache)
        self.pin_models = _RuntimeMirror(self, lambda rt: rt.pin_model)
        self.pin_datas = _RuntimeMirror(self, lambda rt: rt.pin_data)
        self.active_traj: Dict[str, Any] = {}
        self._last_safe_q: Dict[str, np.ndarray] = {}
        self._last_known_q: Dict[str, np.ndarray] = {}
        self._sync_legacy_views()

        self.command_queue: queue.Queue[Command] = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.scene = None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _initialize_runtime(self, runtime: RobotRuntime) -> None:
        runtime.mode = ControlMode.JOINT
        runtime.collision_checker = collision.create_collision_checker(
            runtime.config, self.config, self.logger
        )
        runtime.state_valid_cache = None

        # Build IK cache
        try:
            base_link = "meca_base_link" if "meca" in runtime.config.urdf.lower() else "base_link"
            runtime.drake_cache = DrakeIKCache(
                runtime.config.urdf,
                base_link=base_link,
                ee_link=runtime.config.end_effector_link,
            )
            self.logger.info(f"Loaded Drake IK cache for {runtime.name}")
        except Exception as exc:
            self.logger.warning(f"Failed to load Drake IK cache for {runtime.name}: {exc}")
            runtime.drake_cache = None

        # Pinocchio models for collision
        try:
            import pinocchio as pin

            mdl = pin.buildModelFromUrdf(runtime.config.urdf)
            dat = mdl.createData()
            runtime.pin_model = mdl
            runtime.pin_data = dat
        except Exception as exc:
            self.logger.warning(f"Pinocchio unavailable for {runtime.name}: {exc}")
            runtime.pin_model = None
            runtime.pin_data = None

        self._ik_sanity_check(runtime)

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------
    def build_scene(self) -> None:
        self.scene = scene_builder.build_scene(self.renderer, self.config, self.robots, self.logger)
        self._sync_legacy_views()

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.thread.start()
        self.logger.info("Movement controller started")

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.logger.info("Movement controller stopped")

    def _control_loop(self) -> None:
        dt = self.config.scene.dt
        while self.running:
            start_time = time.time()
            try:
                while True:
                    cmd = self.command_queue.get_nowait()
                    self._process_command(cmd)
            except queue.Empty:
                pass

            self._update_robots()

            if self.scene:
                try:
                    self.scene.step()
                except Exception as exc:
                    self.logger.error(f"Scene step failed: {exc}")
                    break
                self._validate_pending_poses()

            elapsed = time.time() - start_time
            sleep_time = max(0.0, dt - elapsed)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Robot state helpers
    # ------------------------------------------------------------------
    def _get_runtime(self, robot: str) -> RobotRuntime:
        if robot not in self.robots:
            raise KeyError(f"Unknown robot '{robot}'")
        return self.robots[robot]

    def _read_robot_joints(self, runtime: RobotRuntime, prefer_struct: bool = False) -> np.ndarray:
        if runtime.entity is None:
            return np.zeros(0, dtype=np.float32)
        return get_robot_joints(runtime.entity, self.logger, prefer_struct=prefer_struct)

    def _update_all_robot_states(self) -> None:
        for runtime in self.robots.values():
            if runtime.entity is None:
                continue
            try:
                q_current = self._read_robot_joints(runtime, prefer_struct=True)
                if q_current is not None and q_current.size > 0 and np.all(np.isfinite(q_current)):
                    runtime.last_known_q = np.array(q_current, dtype=np.float64)
                    self.logger.debug(f"Updated state for {runtime.name}: {q_current}")
            except Exception as exc:
                self.logger.debug(f"Failed to update state for {runtime.name}: {exc}")
        self._sync_legacy_views()

    def _get_last_known_state(self, robot: str) -> Optional[np.ndarray]:
        runtime = self._get_runtime(robot)
        if runtime.entity is not None:
            try:
                q = self._read_robot_joints(runtime, prefer_struct=True)
                if q.size > 0 and np.all(np.isfinite(q)):
                    runtime.last_known_q = np.array(q, dtype=np.float64)
                    return runtime.last_known_q
            except Exception:
                pass
        return runtime.last_known_q

    def _sync_legacy_views(self) -> None:
        self.robot_modes.refresh()
        self.joint_targets.refresh()
        self.robot_entities.refresh()
        self.collision_checkers.refresh()
        self.drake_caches.refresh()
        self.pin_models.refresh()
        self.pin_datas.refresh()
        self.active_traj = {}
        for name, runtime in self.robots.items():
            if runtime.active_traj:
                self.active_traj[name] = {
                    "waypoints": runtime.active_traj.waypoints,
                    "times": runtime.active_traj.times,
                    "start_t": runtime.active_traj.start_time,
                }
            if runtime.last_safe_q is not None:
                self._last_safe_q[name] = runtime.last_safe_q
            else:
                self._last_safe_q.pop(name, None)
            if runtime.last_known_q is not None:
                self._last_known_q[name] = runtime.last_known_q
            else:
                self._last_known_q.pop(name, None)

    # ------------------------------------------------------------------
    # IK sanity check
    # ------------------------------------------------------------------
    def _ik_sanity_check(self, runtime: RobotRuntime) -> None:
        if runtime.drake_cache is None:
            return
        cache = runtime.drake_cache
        q0 = np.array([np.deg2rad(d) for d in runtime.joint_targets], dtype=np.float64)
        q0 = cache.clamp(q0)
        plant_context = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(plant_context, q0)
        ee_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
        pos = ee_pose.translation()
        rot = ee_pose.rotation()
        quat = rot.ToQuaternion()
        quat_wxyz = (float(quat.w()), float(quat.x()), float(quat.y()), float(quat.z()))
        q_sol, info = solve_ik_drake(
            cache,
            q_seed=q0,
            target_pos_base_m=tuple(pos),
            target_quat_base_wxyz=quat_wxyz,
            is_state_valid=lambda q: True,
            opts=DrakeIKOptions(
                pos_tolerance_m=1e-3,
                rot_tolerance_deg=180.0,
                max_random_seeds=0,
                center_bias_weight=0,
                seed_stick_weight=0,
            ),
        )
        if q_sol is None:
            self.logger.error(f"IK self-test FAILED for {runtime.name}: {info}")
        else:
            self.logger.info(f"IK self-test OK for {runtime.name} [drake_ik]")

    # ------------------------------------------------------------------
    # Command processing
    # ------------------------------------------------------------------
    def _process_command(self, cmd: Command) -> None:
        if isinstance(cmd, SetJointCommand):
            runtime = self._get_runtime(cmd.robot)
            if cmd.joint_idx < len(runtime.joint_targets):
                runtime.joint_targets[cmd.joint_idx] = cmd.value_deg
        elif isinstance(cmd, SetJointTargetsCommand):
            runtime = self._get_runtime(cmd.robot)
            runtime.joint_targets = list(cmd.values_deg)
        elif isinstance(cmd, CartesianMoveCommand):
            runtime = self._get_runtime(cmd.robot)
            self._execute_cartesian_move(runtime, cmd)
        elif isinstance(cmd, SwitchModeCommand):
            runtime = self._get_runtime(cmd.robot)
            runtime.mode = cmd.mode
            self.logger.info(f"Switched {cmd.robot} to {cmd.mode.value} mode")
        self._sync_legacy_views()

    # ------------------------------------------------------------------
    # Cartesian motion planning
    # ------------------------------------------------------------------
    def _build_state_valid(self, runtime: RobotRuntime):
        runtime.state_valid_cache = collision.make_state_valid_fn(
            runtime, self.robots, self.logger, self._get_last_known_state
        )
        return runtime.state_valid_cache

    def _execute_cartesian_move(self, runtime: RobotRuntime, cmd: CartesianMoveCommand) -> None:
        if runtime.drake_cache is None:
            self.logger.error(f"No IK cache for robot {runtime.name}")
            return

        def read_actual_joints() -> np.ndarray:
            return self._read_robot_joints(runtime, prefer_struct=True)

        runtime.last_target_pose = None
        runtime.last_planned_q = None
        runtime.pose_refine_attempts = 0
        plan = plan_cartesian_move(
            runtime,
            cmd,
            self.config,
            self.logger,
            read_actual_joints,
            self._update_all_robot_states,
            lambda: runtime.state_valid_cache or self._build_state_valid(runtime),
        )
        if plan is None:
            return

        waypoints, times = plan
        runtime.active_traj = ActiveTrajectory(waypoints=waypoints, times=times, start_time=time.time())
        self.logger.info(
            f"Cartesian move planned: {runtime.name} ({waypoints.shape[0]} waypoints, {times[-1]:.2f}s duration)"
        )
        self._sync_legacy_views()

    # ------------------------------------------------------------------
    # Robot updates
    # ------------------------------------------------------------------
    def _update_robots(self) -> None:
        now = time.time()
        for runtime in self.robots.values():
            entity = runtime.entity
            if entity is None:
                continue

            if runtime.active_traj:
                sample = runtime.active_traj.sample(now)
                q = sample["q"]
                set_robot_joints(entity, q.tolist(), len(q), self.logger)
                runtime.joint_targets = rad_to_deg_list(q)
                if sample["done"]:
                    runtime.active_traj = None
                    self._record_safe_state(runtime, q)
                    runtime.pending_pose_validation = True
                    runtime.pose_refine_attempts = 0
                continue

            if runtime.joint_targets:
                q_rad = deg_to_rad_list(runtime.joint_targets)
                expected = len(runtime.joint_targets)
                set_robot_joints(entity, q_rad, expected, self.logger)
                self._record_safe_state(runtime, q_rad)

        self._update_all_robot_states()

    def _record_safe_state(self, runtime: RobotRuntime, q) -> None:
        checker = runtime.collision_checker
        mdl = runtime.pin_model
        dat = runtime.pin_data
        if checker and mdl and dat:
            try:
                q_array = np.array(q, dtype=np.float64)
                if not checker.in_collision_from_pin(mdl, dat, q_array):
                    runtime.last_safe_q = q_array.copy()
            except Exception:
                pass

    def _validate_pending_poses(self) -> None:
        for runtime in self.robots.values():
            if not runtime.pending_pose_validation:
                continue
            runtime.pending_pose_validation = self._check_pose(runtime)

    def _check_pose(self, runtime: RobotRuntime) -> bool:
        cache = runtime.drake_cache
        if cache is None or runtime.entity is None:
            return False
        ctrl = self.config.control_for(runtime.name)
        try:
            actual = self._read_robot_joints(runtime)
            self.logger.info(f"ACTUAL GENESIS JOINTS: {runtime.name} = {actual}")
            if actual.size >= 6:
                plant_context = cache.plant.CreateDefaultContext()
                padded = np.zeros(cache.plant.num_positions())
                padded[: actual.size] = actual
                cache.plant.SetPositions(plant_context, padded)
                pose = cache.plant.CalcRelativeTransform(
                    plant_context, cache.base_frame, cache.ee_frame
                )
                pos = pose.translation()
                rot = pose.rotation()
                rpy = rot.ToRollPitchYaw()
                rpy_deg = (
                    np.rad2deg(rpy.roll_angle()),
                    np.rad2deg(rpy.pitch_angle()),
                    np.rad2deg(rpy.yaw_angle()),
                )
                self.logger.info(
                    f"ACTUAL FK POSE: {runtime.name} pos={tuple(pos)} RPY={rpy_deg}"
                )

                if runtime.last_target_pose is not None:
                    target_pos, target_quat = runtime.last_target_pose
                    pos_diff = np.asarray(pos) - target_pos
                    pos_l2 = float(np.linalg.norm(pos_diff))
                    pos_linf = float(np.max(np.abs(pos_diff)))
                    R_ach = rot.matrix()
                    R_des = quaternion_to_rotation_matrix(target_quat)
                    R_err = R_ach.T @ R_des
                    cos_theta = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
                    ang_err_deg = float(np.degrees(np.arccos(cos_theta)))

                    eps = 1e-6
                    if pos_linf > ctrl.ik_pos_tolerance_m + eps or ang_err_deg > ctrl.ik_rot_tolerance_deg + eps:
                        refine_result = None
                        if runtime.pose_refine_attempts < ctrl.ik_refine_max_attempts:
                            runtime.pose_refine_attempts += 1
                            refine_result = self._refine_pose(runtime, actual, target_pos, target_quat)
                        if refine_result is not None:
                            pos_linf_ref, pos_l2_ref, ang_err_ref = refine_result
                            if (
                                pos_linf_ref <= ctrl.ik_pos_tolerance_m + eps
                                and ang_err_ref <= ctrl.ik_rot_tolerance_deg + eps
                            ):
                                runtime.pending_pose_validation = False
                                runtime.pose_refine_attempts = 0
                                self.logger.info(
                                    f"{runtime.name} pose refine succeeded: pos_linf={pos_linf_ref*1000:.3f}mm, "
                                    f"pos_l2={pos_l2_ref*1000:.3f}mm, ang_err={ang_err_ref:.3f}°"
                                )
                                return False
                            else:
                                self.logger.warning(
                                    f"{runtime.name} pose refine still out of bounds: "
                                    f"pos_linf={pos_linf_ref*1000:.3f}mm (limit {ctrl.ik_pos_tolerance_m*1000:.3f}mm), "
                                    f"ang_err={ang_err_ref:.3f}°"
                                )
                                return True
                        runtime.pose_refine_attempts = 0
                        if runtime.last_safe_q is not None:
                            self.logger.warning(
                                f"{runtime.name} pose error exceeds tolerance: "
                                f"pos_linf={pos_linf*1000:.3f}mm (limit {ctrl.ik_pos_tolerance_m*1000:.3f}mm), "
                                f"pos_l2={pos_l2*1000:.3f}mm, "
                                f"ang_err={ang_err_deg:.3f}° (limit {ctrl.ik_rot_tolerance_deg:.3f}°)"
                            )
                            set_robot_joints(runtime.entity, runtime.last_safe_q.tolist(), len(runtime.last_safe_q), self.logger)
                            runtime.joint_targets = rad_to_deg_list(runtime.last_safe_q)
                            runtime.pending_pose_validation = False
                        else:
                            self.logger.warning(
                                f"{runtime.name} pose error exceeds tolerance but no safe state to restore"
                            )
                            runtime.pending_pose_validation = False
                        return False
                    else:
                        self.logger.info(
                            f"{runtime.name} pose error: pos_linf={pos_linf*1000:.3f}mm, "
                            f"pos_l2={pos_l2*1000:.3f}mm, ang_err={ang_err_deg:.3f}°"
                        )
                        runtime.pose_refine_attempts = 0

                if runtime.last_planned_q is not None and runtime.last_planned_q.size > 0:
                    planned_q = runtime.last_planned_q
                    n = min(planned_q.size, actual.size)
                    joint_err = float(np.linalg.norm(actual[:n] - planned_q[:n]))
                    joint_tol = 5e-3  # ~0.29°
                    if joint_err > joint_tol:
                        if runtime.last_safe_q is not None:
                            self.logger.warning(
                                f"{runtime.name} joint deviation from plan = {joint_err:.4f} rad; restoring last safe configuration"
                            )
                            set_robot_joints(runtime.entity, runtime.last_safe_q.tolist(), len(runtime.last_safe_q), self.logger)
                            runtime.joint_targets = rad_to_deg_list(runtime.last_safe_q)
                            runtime.pending_pose_validation = False
                        else:
                            self.logger.warning(
                                f"{runtime.name} joint deviation from plan = {joint_err:.4f} rad; no safe state to restore"
                            )
                            runtime.pending_pose_validation = False
                        return False
                    else:
                        self.logger.debug(
                            f"{runtime.name} joints within {joint_err:.4f} rad of planned goal"
                        )

        except Exception as exc:
            self.logger.error(f"Failed to compute actual FK pose for {runtime.name}: {exc}")
        return False

    def _refine_pose(
        self,
        runtime: RobotRuntime,
        current_q: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> bool:
        cache = runtime.drake_cache
        if cache is None:
            return False

        ctrl = self.config.control_for(runtime.name)
        attempts = max(1, int(ctrl.ik_refine_max_attempts))
        pos_tol = min(float(ctrl.ik_refine_pos_tolerance_m), float(ctrl.ik_pos_tolerance_m))
        rot_tol = min(float(ctrl.ik_refine_rot_tolerance_deg), float(ctrl.ik_rot_tolerance_deg))

        seed = np.array(current_q, dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat = np.asarray(target_quat, dtype=np.float64)
        success_q = None
        for attempt in range(attempts):
            q_ref, info = solve_ik_drake(
                cache,
                q_seed=seed,
                target_pos_base_m=tuple(target_pos),
                target_quat_base_wxyz=tuple(target_quat),
                is_state_valid=lambda q: True,
                opts=DrakeIKOptions(
                    pos_tolerance_m=pos_tol,
                    rot_tolerance_deg=rot_tol,
                    max_random_seeds=4,
                    seed_noise_rad=0.1,
                    center_bias_weight=1e-2,
                    seed_stick_weight=2e-2,
                ),
            )
            if q_ref is not None:
                success_q = q_ref
                break
            if info.get("last_q") is not None:
                seed = np.array(info["last_q"], dtype=np.float64)

        if success_q is None:
            self.logger.warning(f"{runtime.name} IK refine failed after {attempts} attempts")
            return None

        # Evaluate the refined pose using Drake before applying
        plant_context = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(plant_context, success_q)
        pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
        pos = pose.translation()
        rot = pose.rotation()
        pos_diff = pos - target_pos
        pos_linf = float(np.max(np.abs(pos_diff)))
        pos_l2 = float(np.linalg.norm(pos_diff))
        R_err = rot.matrix().T @ quaternion_to_rotation_matrix(target_quat)
        cos_theta = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
        ang_err = float(np.degrees(np.arccos(cos_theta)))

        set_robot_joints(runtime.entity, success_q.tolist(), len(success_q), self.logger)
        runtime.joint_targets = rad_to_deg_list(success_q)
        runtime.last_safe_q = success_q
        runtime.last_planned_q = success_q

        self.logger.info(
            f"{runtime.name} refined pose with tighter tolerances (pos_tol={pos_tol*1000:.2f}mm, rot_tol={rot_tol:.2f}°); "
            f"pos_linf={pos_linf*1000:.3f}mm, ang_err={ang_err:.3f}°"
        )
        return pos_linf, pos_l2, ang_err

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_joint_position(self, robot: str, joint_idx: int, value_deg: float) -> None:
        self.command_queue.put(SetJointCommand(robot, joint_idx, value_deg))

    def set_joint_targets(self, robot: str, values_deg: List[float]) -> None:
        self.command_queue.put(SetJointTargetsCommand(robot, values_deg))

    def move_cartesian(
        self,
        robot: str,
        position: Tuple[float, float, float],
        orientation_deg: Tuple[float, float, float],
        frame: str = "base",
    ) -> None:
        self.command_queue.put(
            CartesianMoveCommand(
                robot=robot,
                position=position,
                orientation_deg=orientation_deg,
                frame=frame,
            )
        )

    def switch_mode(self, robot: str, mode: ControlMode) -> None:
        self.command_queue.put(SwitchModeCommand(robot=robot, mode=mode))

    def get_robot_mode(self, robot: str) -> ControlMode:
        try:
            return self._get_runtime(robot).mode
        except KeyError:
            return ControlMode.JOINT

    def get_joint_targets(self, robot: str) -> List[float]:
        try:
            return list(self._get_runtime(robot).joint_targets)
        except KeyError:
            return []

    def get_joint_positions(self, robot: str) -> List[float]:
        try:
            runtime = self._get_runtime(robot)
        except KeyError:
            return []
        if runtime.entity is None:
            return []
        joints = self._read_robot_joints(runtime, prefer_struct=True)
        return list(rad_to_deg_list(joints))

    def reset_to_last_safe(self, robot: str) -> bool:
        runtime = self._get_runtime(robot)
        if runtime.last_safe_q is not None and runtime.entity is not None:
            set_robot_joints(
                runtime.entity,
                runtime.last_safe_q.tolist(),
                len(runtime.last_safe_q),
                self.logger,
            )
            self.logger.info(f"Reset {robot} to last safe configuration")
            self._sync_legacy_views()
            return True
        self.logger.warning(f"No safe configuration available for {robot}")
        return False


__all__ = ["MovementController"]
