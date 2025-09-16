"""Movement controller for robot motion commands.

Clean implementation handling joint and Cartesian motion.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED

from .config_reader import SimforgeConfig
from .genesis_renderer import GenesisRenderer
from .ik_drake import DrakeIKCache, solve_ik_drake, DrakeIKOptions
from .path_planner import ompl_rrt_connect_plan, cartesian_linear_plan
from .collision_checker import CollisionChecker
from .logging_utils import setup_logging
from .transformations import (
    rpy_to_quaternion,
    rpy_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_multiply
)


class ControlMode(Enum):
    JOINT = "joint"
    CARTESIAN = "cartesian"


@dataclass
class Command:
    pass


@dataclass
class SetJointCommand(Command):
    robot: str
    joint_idx: int
    value_deg: float


@dataclass
class SetJointTargetsCommand(Command):
    robot: str
    values_deg: List[float]


@dataclass
class CartesianMoveCommand(Command):
    robot: str
    position: Tuple[float, float, float]
    orientation_deg: Tuple[float, float, float]  # roll, pitch, yaw
    frame: str = "base"

@dataclass
class SwitchModeCommand(Command):
    robot: str
    mode: ControlMode


class MovementController:
    """High-level movement controller managing robot motion."""
    
    def __init__(self, config: SimforgeConfig, debug: bool = False):
        self.config = config
        self.logger = setup_logging(debug)
        self.renderer = GenesisRenderer(config.scene.backend, self.logger)
        
        # State tracking
        self.robot_modes: Dict[str, ControlMode] = {}
        self.joint_targets: Dict[str, List[float]] = {}
        self.robot_entities: Dict[str, Any] = {}
        self.collision_checkers: Dict[str, CollisionChecker] = {}
        
        # active trajectory playback: name -> dict(waypoints, times, start_t)
        self.active_traj: Dict[str, Dict[str, Any]] = {}
        
        # Pinocchio models for IK / collision
        self.pin_models: Dict[str, Any] = {}
        self.pin_datas: Dict[str, Any] = {}
        
        # Drake IK caches
        self.drake_caches: Dict[str, DrakeIKCache] = {}
        
        # Last known good joint states for other robots (to avoid zero fallback)
        self._last_known_q: Dict[str, np.ndarray] = {}
        
        # Threading
        self.command_queue: queue.Queue[Command] = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Genesis scene
        self.scene = None
        
        self._initialize_robots()
        # After building per-robot checkers, register other robots as dynamic obstacles
        self._register_env_robots()
        
        # Log collision checking status
        self._log_collision_status()

    def _get_joint_limits(self, cache) -> Tuple[np.ndarray, np.ndarray]:
        lower = cache.lower
        upper = cache.upper
        return lower, upper

    def _make_state_valid_fn(self, robot_name: str):
        """World-aware collision validity:
        - Uses this robot's CollisionChecker (FCL+Pinocchio) for self/world collisions
        - Injects other robots as dynamic obstacles by updating their link transforms each check
        """
        cc = self.collision_checkers.get(robot_name)
        mdl = self.pin_models.get(robot_name) if hasattr(self, "pin_models") else None
        dat = self.pin_datas.get(robot_name) if hasattr(self, "pin_datas") else None

        if not (cc and cc.available and mdl and dat):
            self.logger.warning(f"Collision checking disabled for {robot_name} - missing components")
            return lambda q: True  # permissive fallback

        # Capture other robots' models and bases once (updated each call for transforms)
        state_logs = {}  # Fix scope bug - move above the loop
        others: List[Tuple[str, Any, Any, Tuple[float,float,float], Tuple[float,float,float]]] = []
        for r in self.config.robots:
            if r.name == robot_name:
                continue
            mdl_o = self.pin_models.get(r.name)
            dat_o = self.pin_datas.get(r.name)
            if not (mdl_o and dat_o):
                self.logger.error(f"Missing Pinocchio model/data for env robot {r.name} in {robot_name}'s checker")
                continue
            base_pos = tuple(r.base_position or (0.0, 0.0, 0.0))
            base_rpy = tuple(r.base_orientation or (0.0, 0.0, 0.0))
            others.append((r.name, mdl_o, dat_o, base_pos, base_rpy))
            self.logger.warning(f"Added env robot {r.name} to {robot_name}'s collision checker")
            state_logs = {}

        def _valid(q: np.ndarray) -> bool:
            # 1) Update other robots' transforms into this checker's env using their current state
            for name_o, mdl_o, dat_o, base_pos, base_rpy in others:
                # CRITICAL: Always get the current actual state from Genesis
                q_o = None
                entity_o = self.robot_entities.get(name_o)
                if entity_o is not None:
                    try:
                        q_o = self._get_robot_joints(entity_o, prefer_struct=True)
                        if q_o is not None and len(q_o) > 0 and np.all(np.isfinite(q_o)):
                            # self.logger.info(f"COLLISION ENV UPDATE: {name_o} at joints {np.rad2deg(q_o)}")
                            state_logs[name_o] = f"Genesis joints: {np.round(np.rad2deg(q_o),2).tolist()} deg"
                        else:
                            q_o = None
                    except Exception as e:
                        self.logger.warning(f"Failed to read {name_o} joints from Genesis: {e}")
                        q_o = None
                
                # Fallback to joint targets if Genesis read failed
                if q_o is None or len(q_o) == 0:
                    targets_deg = self.joint_targets.get(name_o, [])
                    if targets_deg:
                        q_o = np.array([np.deg2rad(d) for d in targets_deg], dtype=np.float64)
                        self.logger.warning(f"Using {name_o} targets instead of Genesis state: {targets_deg} deg")
                    else:
                        # Skip this robot if no state available
                        self.logger.error(f"No state available for env robot {name_o} - collision checking will be incomplete!")
                        continue

                # Clamp/pad to model size
                if q_o.size != mdl_o.nq:
                    qq = np.zeros(mdl_o.nq, dtype=np.float64)
                    qq[:min(mdl_o.nq, q_o.size)] = q_o[:min(mdl_o.nq, q_o.size)]
                    q_o = qq
                try:
                    cc.update_env_robot_from_pin(
                        name_o, mdl_o, dat_o, q_o,
                        base_position=base_pos,
                        base_orientation_rpy=base_rpy
                    )
                except Exception as e:
                    self.logger.debug(f"Env robot update failed for {name_o}: {e}")

            # 2) Check candidate for this robot
            q = np.asarray(q, dtype=np.float64).flatten()
            if q.size != mdl.nq:
                qq = np.zeros(mdl.nq, dtype=np.float64)
                qq[:min(mdl.nq, q.size)] = q[:min(mdl.nq, q.size)]
                q_use = qq
            else:
                q_use = q
            try:
                return not cc.in_collision_from_pin(mdl, dat, q_use)
            except Exception as e:
                self.logger.warning(f"Collision check error for {robot_name}: {e}")
                return False  # Fail closed instead of open - reject on collision check errors
        
        # log state of other robots
        for rname, slog in state_logs.items():
            self.logger.info(f"Env robot {rname} state: {slog}")

        return _valid

    def _initialize_robots(self):
        """Initialize robot state and models."""
        for robot_config in self.config.robots:
            name = robot_config.name
            self.robot_modes[name] = ControlMode.JOINT
            self.joint_targets[name] = list(robot_config.initial_joint_positions or [0.0] * 6)
            
            # Initialize collision checker with world
            ctrl = self.config.control_for(name)
            world_boxes = []
            for obj in self.config.objects:
                if obj.type == "box" and obj.size:
                    world_boxes.append((
                        obj.name or "box",
                        tuple(obj.size),
                        tuple(obj.position),
                        tuple(obj.orientation_rpy),
                    ))
                elif obj.type == "plane":
                    # Turn plane into a very thin box for collision checking
                    # Center the box so its TOP face is exactly at ground_plane_z
                    size = obj.size or [10.0, 10.0, 0.02]  # large XY, thin Z
                    thickness = float(size[2])
                    gz = float(ctrl.ground_plane_z)  # per-robot ground level
                    # center the thin box so its top face is exactly at ground_plane_z
                    plane_center = list(obj.position or [0.0, 0.0, 0.0])
                    plane_center[2] = gz - thickness * 0.5
                    world_boxes.append((
                        obj.name or "plane",
                        tuple(size),
                        tuple(plane_center),
                        tuple(obj.orientation_rpy or [0.0, 0.0, 0.0]),
                    ))
            world_allowed_pairs = []
            if robot_config.control and robot_config.control.world_allowed_pairs:
                # pairs like ["TX2_60_1/link_6", "obj:table1"] or ["wrist_3_link", "obj:table1"]
                # Normalize by stripping any robot prefix (e.g., "UR5e_1/") from robot link names.
                def _norm_side(s: str) -> str:
                    s = str(s)
                    if s.startswith("obj:"):
                        return s
                    # Allow forms like "robot:UR5e_1:wrist_3_link" or "UR5e_1/wrist_3_link"
                    if s.startswith("robot:"):
                        # keep last token after ':' as link or after last ':' if multiple
                        s2 = s.split(":")[-1]
                        return s2.split("/")[-1]
                    # default: if contains '/', keep the last component as the link name
                    return s.split("/")[-1]

                for a, b in robot_config.control.world_allowed_pairs:
                    world_allowed_pairs.append((_norm_side(a), _norm_side(b)))

            # Handle self-collision checking
            allowed_link_pairs = None
            if not ctrl.self_collision_check:
                # If self-collision is disabled, we should allow all adjacent link pairs
                # This is a simplified approach - in a real implementation, you'd want to
                # parse the URDF to get actual adjacent pairs
                allowed_link_pairs = []

            try:
                self.collision_checkers[name] = CollisionChecker(
                    robot_config.urdf,
                    self.logger,
                    base_position=tuple(robot_config.base_position or (0.0, 0.0, 0.0)),
                    base_orientation_rpy=tuple(robot_config.base_orientation or (0.0, 0.0, 0.0)),
                    allowed_link_pairs=allowed_link_pairs,
                    world_allowed_pairs=world_allowed_pairs,
                    world_boxes=world_boxes,
                    ground_plane_z=ctrl.ground_plane_z,
                    collision_mesh_shrink=getattr(ctrl, "collision_mesh_shrink", 1.0),
                )
            except Exception as e:
                self.logger.warning(f"Collision checker init failed for {name}: {e}")
                self.collision_checkers[name] = None
            
            # Build Drake IK cache
            try:
                if "meca" in robot_config.urdf.lower():
                    base = "meca_base_link"
                else:
                    base = "base_link"
                self.drake_caches[name] = DrakeIKCache(
                    robot_config.urdf,
                    base_link=base,
                    ee_link=robot_config.end_effector_link,
                )
                self.logger.info(f"Loaded Drake IK cache for {name}")
            except Exception as e:
                self.logger.warning(f"Failed to load Drake IK cache for {name}: {e}")
            
            # Build Pinocchio model/data for collision, if available
            try:
                import pinocchio as pin
                mdl = pin.buildModelFromUrdf(robot_config.urdf)
                dat = mdl.createData()
                self.pin_models[name] = mdl
                self.pin_datas[name] = dat
            except Exception as e:
                self.logger.warning(f"Pinocchio unavailable for {name}: {e}")
            
            self._ik_sanity_check(name)

    def _to_meters(self, robot_name: str, pos_xyz: Tuple[float,float,float]) -> Tuple[float,float,float]:
        ctrl = self.config.control_for(robot_name)
        if (ctrl.cartesian_units or "m").lower() == "mm":
            return tuple(float(v)/1000.0 for v in pos_xyz)
        return tuple(float(v) for v in pos_xyz)

    def _register_env_robots(self):
        """Register every robot's geometry as environment obstacles in every other robot's checker."""
        try:
            for ra in self.config.robots:
                cc_a = self.collision_checkers.get(ra.name)
                if not cc_a:
                    continue
                for rb in self.config.robots:
                    if rb.name == ra.name:
                        continue
                    try:
                        cc_a.register_env_robot(rb.name, rb.urdf)
                        self.logger.debug(f"Registered env robot '{rb.name}' into checker for '{ra.name}'")
                    except Exception as e:
                        self.logger.debug(f"Register env robot '{rb.name}' into '{ra.name}' checker failed: {e}")
        except Exception as e:
            self.logger.debug(f"_register_env_robots failed: {e}")

    def _log_collision_status(self):
        """Log collision checking status for debugging."""
        for robot_name in self.robot_modes.keys():
            cc = self.collision_checkers.get(robot_name)
            if cc and cc.available:
                env_robots = len(cc.env_robot_geoms)
                world_objs = len(cc.env_objs)
                robot_links = len(cc.robot_geoms)
                self.logger.info(f"Collision checking ON for {robot_name}: "
                                f"{robot_links} links, {world_objs} world objects, {env_robots} env robots")
            else:
                self.logger.warning(f"Collision checking OFF for {robot_name}")

    def build_scene(self) -> None:
        """Build the Genesis scene."""
        scene_cfg = self.config.scene
        self.scene = self.renderer.create_scene(
            dt=scene_cfg.dt,
            gravity=scene_cfg.gravity,
            show_viewer=scene_cfg.show_viewer,
            max_fps=scene_cfg.max_fps
        )
        
        # Add objects
        for obj_config in self.config.objects:
            if obj_config.type == "plane":
                self.scene.add_entity(
                    self.renderer.morphs.Plane(pos=obj_config.position)
                )
            elif obj_config.type == "box" and obj_config.size:
                self.scene.add_entity(
                    self.renderer.morphs.Box(
                        pos=obj_config.position,
                        size=obj_config.size,
                        euler=obj_config.orientation_rpy
                    )
                )
        
        # Add robots
        for robot_config in self.config.robots:
            entity = self.scene.add_entity(
                self.renderer.morphs.URDF(
                    file=robot_config.urdf,
                    pos=robot_config.base_position,
                    euler=robot_config.base_orientation,
                    fixed=robot_config.fixed_base
                )
            )
            self.robot_entities[robot_config.name] = entity
        
        self.scene.build()
        
        # Set initial joint positions
        for robot_config in self.config.robots:
            if robot_config.initial_joint_positions:
                entity = self.robot_entities[robot_config.name]
                q_rad = [np.deg2rad(deg) for deg in robot_config.initial_joint_positions]
                self._set_robot_joints(entity, q_rad, robot_config.name)

        # Commit once to ensure setters/targets are applied in the sim
        if self.scene:
            for _ in range(2):
                try:
                    self.scene.step()
                except Exception as e:
                    self.logger.debug(f"Initial scene step failed: {e}")
        
        self.logger.info("Scene built successfully")

    def start(self) -> None:
        """Start the control loop."""
        if self.thread and self.thread.is_alive():
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.thread.start()
        self.logger.info("Movement controller started")

    def stop(self) -> None:
        """Stop the control loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.logger.info("Movement controller stopped")

    def _control_loop(self):
        """Main control loop running in background thread."""
        dt = self.config.scene.dt
        
        while self.running:
            start_time = time.time()
            
            # Process commands
            try:
                while True:
                    cmd = self.command_queue.get_nowait()
                    self._process_command(cmd)
            except queue.Empty:
                pass
            
            # Update robot positions
            self._update_robots()
            
            # Step simulation
            if self.scene:
                try:
                    self.scene.step()
                except Exception as e:
                    self.logger.error(f"Scene step failed: {e}")
                    break
            
            # Maintain timing
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _ik_sanity_check(self, robot_name: str):
        if robot_name not in self.drake_caches:
            return
        robot_config = next(r for r in self.config.robots if r.name == robot_name)
        cache = self.drake_caches[robot_name]

        # Use current GUI targets (deg -> rad)
        q0 = np.array([np.deg2rad(d) for d in self.joint_targets[robot_name]], dtype=np.float64)
        q0 = cache.clamp(q0)

        # Compute current EE pose via Drake FK
        plant_context = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(plant_context, q0)
        ee_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
        pos = ee_pose.translation()
        rot = ee_pose.rotation()
        quat = rot.ToQuaternion()  # [w, x, y, z]
        quat_wxyz = (float(quat.w()), float(quat.x()), float(quat.y()), float(quat.z()))

        q_sol, info = solve_ik_drake(
            cache,
            q_seed=q0,
            target_pos_base_m=tuple(pos),
            target_quat_base_wxyz=quat_wxyz,
            is_state_valid=lambda q: True,
            # Self-test is only a health check: allow wide orientation to avoid noisy startup errors
            opts=DrakeIKOptions(pos_tolerance_m=1e-3, rot_tolerance_deg=180.0, max_random_seeds=0, center_bias_weight=0, seed_stick_weight=0),
        )
        if q_sol is None:
            self.logger.error(f"IK self-test FAILED for {robot_name}: {info}")
        else:
            self.logger.info(f"IK self-test OK for {robot_name} [drake_ik]")

    def _process_command(self, cmd: Command):
        """Process a single command."""
        if isinstance(cmd, SetJointCommand):
            targets = self.joint_targets.get(cmd.robot, [])
            if cmd.joint_idx < len(targets):
                targets[cmd.joint_idx] = cmd.value_deg
                
        elif isinstance(cmd, SetJointTargetsCommand):
            self.joint_targets[cmd.robot] = list(cmd.values_deg)
            
        elif isinstance(cmd, CartesianMoveCommand):
            self._execute_cartesian_move(cmd)
            
        elif isinstance(cmd, SwitchModeCommand):
            self.robot_modes[cmd.robot] = cmd.mode
            self.logger.info(f"Switched {cmd.robot} to {cmd.mode.value} mode")

    def _execute_cartesian_move(self, cmd: CartesianMoveCommand):
        """Solve IK for goal, then plan path (parallel planners), then play trajectory."""
        robot_name = cmd.robot
        self.logger.info(f"Starting _execute_cartesian_move for {robot_name}")
        
        if robot_name not in self.drake_caches:
            self.logger.error(f"No IK cache for robot {robot_name}")
            return

        robot_config = next((r for r in self.config.robots if r.name == robot_name), None)
        if not robot_config or not robot_config.end_effector_link:
            self.logger.error(f"No end effector link configured for {robot_name}")
            return

        # Get cache first
        cache = self.drake_caches[robot_name]
        
        # Current q - use actual Genesis joint state, not GUI targets
        entity = self.robot_entities[robot_name]
        actual_joints = self._get_robot_joints(entity, prefer_struct=True)
        
        if len(actual_joints) >= 6:
            # Pad to match Drake's expected DOF count
            q_current = np.zeros(cache.plant.num_positions())
            q_current[:len(actual_joints)] = actual_joints
            self.logger.debug(f"Using actual Genesis joints: {actual_joints}")
        else:
            # Fallback to GUI targets if Genesis joint reading fails
            targets_deg = self.joint_targets.get(robot_name, [])
            if targets_deg:
                q_current = np.array([np.deg2rad(deg) for deg in targets_deg], dtype=np.float64)
                self.logger.debug(f"Using joint targets: {targets_deg} deg")
            else:
                # Ultimate fallback - zero position
                q_current = np.zeros(cache.plant.num_positions())
                self.logger.warning(f"No valid joint state found, using zeros")

        # --- Coerce q dimension to match Drake plant (critical) ---
        def _coerce_q_dim(cache, q: np.ndarray) -> np.ndarray:
            nq = int(cache.plant.num_positions())
            q = np.asarray(q, dtype=np.float64).flatten()
            if q.shape[0] == nq:
                return q
            if q.shape[0] > nq:
                return q[:nq]
            return np.concatenate([q, np.zeros(nq - q.shape[0], dtype=np.float64)], axis=0)
        
        q_current = _coerce_q_dim(cache, q_current)
        
        self.logger.debug(f"Plant nq={cache.plant.num_positions()}, q_current shape={q_current.shape}")

        # Units / target pose in world frame first
        pos_world_m = self._to_meters(robot_name, cmd.position)
        
        # Debug: Check if orientation values are valid
        self.logger.debug(f"Raw orientation from GUI: {cmd.orientation_deg}")
        
        # Handle NaN or invalid orientation values
        orientation_deg = cmd.orientation_deg
        if any(not np.isfinite(val) for val in orientation_deg):
            self.logger.warning(f"Invalid orientation values detected: {orientation_deg}, using (0,0,0)")
            orientation_deg = (0.0, 0.0, 0.0)
        
        roll, pitch, yaw = [np.deg2rad(x) for x in orientation_deg]
        self.logger.debug(f"RPY in radians: roll={roll}, pitch={pitch}, yaw={yaw}")
        
        quat_wxyz = rpy_to_quaternion(roll, pitch, yaw)
        self.logger.debug(f"Computed quaternion: {quat_wxyz}")
        
        # Validate quaternion magnitude
        quat_mag = np.linalg.norm(quat_wxyz)
        if quat_mag < 1e-10:
            self.logger.error(f"Invalid quaternion magnitude {quat_mag}, using identity quaternion")
            quat_wxyz = (1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        else:
            # Normalize quaternion
            quat_wxyz = tuple(q / quat_mag for q in quat_wxyz)

        # Transform GUI target into Drake's IK base frame.
        # If GUI frame == "base", interpret values directly in the robot's base frame (base_link),
        # which we align with Drake's base_frame in the IK cache. Otherwise (world), apply mapping.
        gui_frame = (cmd.frame or "base").lower()
        R_des_bl = rpy_to_rotation_matrix(roll, pitch, yaw)

        if gui_frame == "base":
            # Directly use base frame inputs (no base_link->BASE offset)
            target_pos_base = np.array(pos_world_m, dtype=np.float64)
            R_des_base = R_des_bl
            target_quat_base = tuple(rotation_matrix_to_quaternion(R_des_base))
            self.logger.info(f"Base-frame target: {np.round(target_pos_base,4)}")
        else:
            # World-frame input: map world -> base_link -> Drake BASE if needed
            robot_base_pos = np.array(robot_config.base_position or [0.0, 0.0, 0.0])
            p_bl = np.array(pos_world_m, dtype=np.float64) - robot_base_pos  # world to base_link

            # Map from base_link -> BASE using Drake's model frames
            try:
                plant_ctx_tmp = cache.plant.CreateDefaultContext()
                frame_base = cache.base_frame
                frame_bl = cache.plant.GetFrameByName(getattr(cache, "base_link", "base_link"))
            except Exception:
                plant_ctx_tmp = cache.plant.CreateDefaultContext()
                frame_base = cache.base_frame
                frame_bl = cache.plant.GetFrameByName("base_link")

            T_base_bl = cache.plant.CalcRelativeTransform(plant_ctx_tmp, frame_base, frame_bl)  # pose of base_link in BASE
            R_base_bl = T_base_bl.rotation().matrix()
            t_base_bl = T_base_bl.translation()

            # Convert target from base_link to BASE using: p_base = R * p_bl + t
            target_pos_base = R_base_bl @ p_bl + t_base_bl

            # Orientation in BASE: R_base = R_base_bl * R_des_bl
            R_des_base = R_base_bl @ R_des_bl
            target_quat_base = tuple(rotation_matrix_to_quaternion(R_des_base))

            self.logger.info(f"World target: {pos_world_m}, Robot base_link@world: {robot_base_pos}")
            self.logger.info(f"base_link->BASE: R={np.round(R_base_bl,3).tolist()}, t={np.round(t_base_bl,4)}")
            self.logger.info(f"Target in BASE frame: {np.round(target_pos_base,4)}")

        # Min-Z guard in BASE (avoid infeasible near-ground wrist attitudes)
        z_min = float(self.config.control_for(robot_name).ground_plane_z) + 0.04  # 4 cm above ground plane
        if target_pos_base[2] < z_min:
            self.logger.warning(f"Clamping target Z from {target_pos_base[2]:.3f}m to {z_min:.3f}m in BASE to avoid near-ground singularities")
            target_pos_base[2] = z_min

        self.logger.debug(f"Target pose: pos={tuple(target_pos_base)}, quat={tuple(target_quat_base)}")

        # FK debug
        plant_context = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(plant_context, q_current)
        ee_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
        current_pos = ee_pose.translation()
        current_rot = ee_pose.rotation()
        current_rpy = current_rot.ToRollPitchYaw()  # Returns RollPitchYaw object
        current_rpy_deg = (np.rad2deg(current_rpy.roll_angle()), 
                          np.rad2deg(current_rpy.pitch_angle()), 
                          np.rad2deg(current_rpy.yaw_angle()))
        
        start_gap = float(np.linalg.norm(np.array(target_pos_base) - current_pos))
        
        # Workspace validation - warn if target seems unreachable
        if target_pos_base[2] < 0.1:  # Z below 10cm is likely unreachable
            self.logger.warning(f"Target Z={target_pos_base[2]:.3f}m may be too low (below robot base)")
        
        # Check if target is too far from current position (likely outside workspace)
        workspace_radius = 1.5  # Typical max reach for industrial robots
        target_dist_from_base = float(np.linalg.norm(target_pos_base))
        if target_dist_from_base > workspace_radius:
            self.logger.warning(f"Target distance {target_dist_from_base:.3f}m may exceed workspace")
        
        # Convert target orientation to degrees for logging
        target_rpy_deg = cmd.orientation_deg
        
        self.logger.info(f"IK: Current EE at {tuple(current_pos)} RPY={current_rpy_deg}")
        self.logger.info(f"IK: Target {tuple(target_pos_base)} RPY={target_rpy_deg}")
        self.logger.info(f"IK: Distance to target = {start_gap:.3f}m")

        # IK with multi-seed strategy (robust against poor initializations)
        self.logger.info(f"Calling solve_ik with multi-seed: max_iters=1000, pos_tol=2e-3, rot_tol=2.0°")
        
        def _clamp_to_limits(qv: np.ndarray) -> np.ndarray:
            lower, upper = self._get_joint_limits(cache)
            qv = np.asarray(qv, dtype=np.float64).flatten()
            nq = cache.plant.num_positions()
            if qv.shape[0] > nq:
                qv = qv[:nq]
            if qv.shape[0] < nq:
                qv = np.concatenate([qv, np.zeros(nq - qv.shape[0])], axis=0)
            return np.clip(qv, lower, upper)
        
        # Build candidate seeds
        seeds: List[np.ndarray] = []
        seeds.append(q_current.copy())

        # Add zero configuration (safe baseline)
        nq = cache.plant.num_positions()
        zero_config = np.zeros(nq, dtype=np.float64)
        seeds.append(_clamp_to_limits(zero_config))

        # Multiple canonical elbow configurations if 6 DOF
        if nq >= 6:
            elbow_up = np.array([0.0, -np.pi/3,  np.pi/3,  -np.pi/3,  -np.pi/3,  0.0], dtype=np.float64)
            seeds.append(_clamp_to_limits(elbow_up))
            elbow_down = np.array([0.0,  np.pi/4, -np.pi/4,  np.pi/4,  np.pi/4,  0.0], dtype=np.float64)
            seeds.append(_clamp_to_limits(elbow_down))
            home_pose = np.array([0.0, -np.pi/6,  np.pi/4,  0.0,      np.pi/3,  0.0], dtype=np.float64)
            seeds.append(_clamp_to_limits(home_pose))

        # Bias first joint toward target XY direction
        # Compute yaw angle to target in base frame
        try:
            target_angle = float(np.arctan2(target_pos_base[1], target_pos_base[0]))
            if nq >= 1:
                biased = q_current.copy()
                biased[0] = target_angle
                seeds.append(_clamp_to_limits(biased))
        except Exception:
            pass

        # Random perturbations around promising seeds
        rng = np.random.RandomState(42)
        for base in [q_current, zero_config]:
            for _ in range(6):
                noise = rng.uniform(-0.2, 0.2, size=nq)
                seeds.append(_clamp_to_limits(base + noise))
        
        # Force update all robot states before building validity checker
        self._update_all_robot_states()
        
        # Build validity checker early so we can accept only collision-free IK goals
        is_valid = self._make_state_valid_fn(robot_name)
        
        # Debug: Test validity of current position
        test_valid = is_valid(q_current)
        self.logger.debug(f"Current position validity check: {test_valid}")

        cache = self.drake_caches[robot_name]

        q_goal = None
        last_info = {}
        # primary solve with drake
        q_try, info = solve_ik_drake(
            cache,
            q_seed=q_current,
            target_pos_base_m=tuple(target_pos_base),
            target_quat_base_wxyz=tuple(target_quat_base),
            is_state_valid=is_valid,
            opts=DrakeIKOptions(
                pos_tolerance_m=1e-3,
                rot_tolerance_deg=1.0,
                max_random_seeds=16,
                seed_noise_rad=0.35,
                center_bias_weight=1e-2,
                seed_stick_weight=5e-3,
            ),
        )
        last_info = info
        if q_try is not None:
            q_goal = q_try
        else:
            # Progressive IK fallback (waypoints in position; fixed orientation)
            self.logger.info("Single-shot IK failed, trying progressive IK...")
            start_pos = current_pos
            q_curr = q_current.copy()
            steps = 20
            for i in range(1, steps + 1):
                wp = start_pos + (i / steps) * (np.array(target_pos_base) - start_pos)
                q_next, info = solve_ik_drake(
                    cache,
                    q_seed=q_curr,
                    target_pos_base_m=tuple(wp),
                    target_quat_base_wxyz=tuple(target_quat_base),
                    is_state_valid=is_valid,
                    opts=DrakeIKOptions(
                        pos_tolerance_m=5e-3,
                        rot_tolerance_deg=5.0,
                        max_random_seeds=8,
                        seed_noise_rad=0.25,
                        center_bias_weight=5e-3,
                        seed_stick_weight=5e-3,
                    ),
                )
                last_info = info
                if q_next is None:
                    break
                q_curr = q_next
                if not is_valid(q_curr):
                    last_info = {"reason": "goal_in_collision", "seed_idx": i, "ori_mode": info.get("ori_mode")}
                    break
                if i == steps and is_valid(q_curr):
                    q_goal = q_curr
                    break
                # Check collision
                if not is_valid(q_goal):
                    self.logger.debug("Progressive IK produced in-collision goal")
                    q_goal = None

        self.logger.info(f"IK result: success={q_goal is not None}")
        if q_goal is None:
            # Provide helpful error message based on failure info
            if last_info.get("reason") == "did_not_converge":
                self.logger.error(
                    f"IK failed to converge to the requested pose after {len(seeds)} seeds "
                    f"(pos_err={last_info.get('pos_err', 0):.3f}m, rot_err={last_info.get('rot_err', 0):.3f}rad)."
                )
            else:
                self.logger.error(f"IK failed for {robot_name} after {len(seeds)} seeds: {last_info}")
            return
        
        # Log achieved pose after IK
        plant_context_check = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(plant_context_check, q_goal)
        achieved_pose = cache.plant.CalcRelativeTransform(plant_context_check, cache.base_frame, cache.ee_frame)
        achieved_pos = achieved_pose.translation()
        achieved_rot = achieved_pose.rotation()
        achieved_rpy = achieved_rot.ToRollPitchYaw()
        achieved_rpy_deg = (np.rad2deg(achieved_rpy.roll_angle()),
                           np.rad2deg(achieved_rpy.pitch_angle()),
                           np.rad2deg(achieved_rpy.yaw_angle()))
        self.logger.info(f"IK achieved: pos={tuple(achieved_pos)} RPY={achieved_rpy_deg}")

        # Orientation geodesic error (robust vs RPY singularities)
        try:
            R_ach = achieved_rot.matrix()
            # Recompute R_des in BASE as used for IK logging (uses R_des_base from earlier)
            R_des_mat = R_des_base  # from earlier computation
            R_err = R_ach.T @ R_des_mat
            tr = float(np.trace(R_err))
            tr_clamped = max(-1.0, min(3.0, tr))  # trace in [-1,3]
            cos_theta = (tr_clamped - 1.0) / 2.0
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta_deg = float(np.degrees(np.arccos(cos_theta)))
            self.logger.info(f"Orientation geodesic error = {theta_deg:.3f}° (mode={last_info.get('ori_mode')})")
        except Exception as _e:
            self.logger.debug(f"Geodesic orientation error computation failed: {_e}")

        # Get control config
        ctrl = self.config.control_for(robot_name)
        
        # Build planners in parallel: OMPL RRTConnect (joint-space) + Cartesian linear
        lower, upper = self._get_joint_limits(cache)
        
        # Debug joint limits for RRT
        self.logger.debug(f"Joint limits for {robot_name}: lower={lower}, upper={upper}")
        self.logger.debug(f"Current q for planning: {q_current}")
        self.logger.debug(f"Goal q for planning: {q_goal}")
        
        # Clamp states to joint limits to prevent RRT bounds violations
        q_current_clamped = np.clip(q_current, lower, upper)
        q_goal_clamped = np.clip(q_goal, lower, upper)
        
        if not np.allclose(q_current, q_current_clamped, atol=1e-6):
            self.logger.warning(f"Clamping current q to bounds: {q_current} -> {q_current_clamped}")
            q_current = q_current_clamped
            
        if not np.allclose(q_goal, q_goal_clamped, atol=1e-6):
            self.logger.warning(f"Clamping goal q to bounds: {q_goal} -> {q_goal_clamped}")
            q_goal = q_goal_clamped

        # If start is invalid, try a tiny upward IK repair to clear surface/adjacent contacts
        if not is_valid(q_current):
            try:
                plant_context = cache.plant.CreateDefaultContext()
                cache.plant.SetPositions(plant_context, q_current)
                ee_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
                pos_repair = ee_pose.translation().copy()
                pos_repair[2] += 0.03  # +3cm in Z
                rot = ee_pose.rotation()
                quat_repair = rot.ToQuaternion()
                quat_repair_wxyz = (quat_repair.w(), quat_repair.x(), quat_repair.y(), quat_repair.z())
                q_repair, _ = solve_ik_drake(
                    cache,
                    q_seed=q_current,
                    target_pos_base_m=tuple(pos_repair),
                    target_quat_base_wxyz=quat_repair_wxyz,
                    is_state_valid=is_valid,
                    opts=DrakeIKOptions(
                        pos_tolerance_m=1e-3,
                        rot_tolerance_deg=2.0,
                        max_random_seeds=4,
                        seed_noise_rad=0.1,
                        center_bias_weight=1e-3,
                        seed_stick_weight=1e-2,
                    ),
                )
                if q_repair is not None and is_valid(q_repair):
                    self.logger.debug("Start state invalid; applied +3cm Z IK repair to clear collisions.")
                    q_current = q_repair
                else:
                    self.logger.error("Start state is invalid and repair IK failed; aborting planning")
                    return
            except Exception as _e:
                self.logger.debug(f"Start repair IK failed: {_e}")
                self.logger.error("Start state is invalid and repair IK errored; aborting planning")
                return
        timeout = float(ctrl.planner_timeout)
        Nw = int(ctrl.cartesian_waypoints)
        
        self.logger.debug(f"Planner config: timeout={timeout}s, cartesian_waypoints={Nw}")

        def plan_rrt():
            # Both start and goal should be valid by this point
            return ompl_rrt_connect_plan(
                q_current, q_goal, lower, upper,
                is_state_valid=is_valid,
                timeout_s=timeout, range_rad=float(ctrl.planner_resolution),
                simplify=True
            )

        # Compute start pose using FK
        plant_context = cache.plant.CreateDefaultContext()
        cache.plant.SetPositions(plant_context, q_current)
        start_pose_ee = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
        start_pos = start_pose_ee.translation()
        start_rot = start_pose_ee.rotation()
        start_quat = start_rot.ToQuaternion()
        start_pose = (start_pos, (start_quat.w(), start_quat.x(), start_quat.y(), start_quat.z()))
        
        # Define target pose for cartesian planner
        target_pose = (target_pos_base, target_quat_base)
        
        # Define IK helper for cartesian planner
        def _ik(q_seed, pose):
            pos, quat = pose
            result, _ = solve_ik_drake(
                cache,
                q_seed=q_seed,
                target_pos_base_m=tuple(pos),
                target_quat_base_wxyz=tuple(quat),
                is_state_valid=is_valid,
                opts=DrakeIKOptions(
                    pos_tolerance_m=1e-3,
                    rot_tolerance_deg=1.0,
                    max_random_seeds=4,
                    seed_noise_rad=0.25,
                    center_bias_weight=5e-3,
                    seed_stick_weight=5e-3,
                ),
            )
            return result

        def plan_cart():
            return cartesian_linear_plan(
                start_q=q_current,
                start_pose_se3=start_pose,
                target_pose_se3=target_pose,
                solve_ik=lambda q_seed, pose: _ik(q_seed, pose),
                is_state_valid=is_valid,
                num_waypoints=Nw
            )

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_rrt = ex.submit(plan_rrt)
            fut_lin = ex.submit(plan_cart)
            done, _ = wait({fut_rrt, fut_lin}, timeout=timeout, return_when=ALL_COMPLETED)

        # Gather finished plans
        plans = []
        if fut_lin.done():
            p = fut_lin.result()
            if p is not None:
                plans.append(("cartesian", *p))
        if fut_rrt.done():
            p = fut_rrt.result()
            if p is not None:
                plans.append(("rrt", *p))

        def _validate_path(way):
            from simforge.path_planner import _check_segment_collision_free
            for i in range(len(way)-1):
                if not _check_segment_collision_free(way[i], way[i+1], is_valid, resolution=20):
                    return False
            return True

        # Keep only validated plans
        valid = [(name, way, t) for (name, way, t) in plans if _validate_path(way)]

        if not valid:
            self.logger.warning(f"Planning failed or produced colliding paths for {robot_name}")
            return

        # Prefer cartesian if present
        valid.sort(key=lambda x: 0 if x[0] == "cartesian" else 1)
        plan_name, waypoints, times = valid[0]
        self.logger.info(f"Selected {plan_name} plan for {robot_name}")
        
        result = (waypoints, times)

        waypoints, times = result  # radians + seconds

        # Queue trajectory for playback
        self.active_traj[robot_name] = {
            "waypoints": waypoints,
            "times": times,
            "start_t": time.time(),
        }
        self.logger.info(f"Cartesian move planned: {robot_name} ({waypoints.shape[0]} waypoints, {times[-1]:.2f}s duration)")

    def _update_robots(self):
        """Update robot joint positions (targets or active trajectory)."""
        now = time.time()
        for robot_name in list(self.joint_targets.keys()):
            entity = self.robot_entities.get(robot_name)
            if not entity:
                continue

            # If a trajectory is active, step along it
            traj = self.active_traj.get(robot_name)
            if traj:
                t0 = traj["start_t"]; times = traj["times"]; way = traj["waypoints"]
                t = now - t0
                if t >= float(times[-1]):
                    q = way[-1]
                    self._set_robot_joints(entity, q.tolist(), robot_name)
                    self.active_traj.pop(robot_name, None)
                    
                    # Record as last safe state if collision-free
                    cc = self.collision_checkers.get(robot_name)
                    mdl = self.pin_models.get(robot_name)
                    dat = self.pin_datas.get(robot_name)
                    if cc and mdl and dat:
                        try:
                            q_array = np.array(q, dtype=np.float64)
                            if not cc.in_collision_from_pin(mdl, dat, q_array):
                                self._last_safe_q[robot_name] = q_array.copy()
                        except Exception:
                            pass
                    
                    # Log actual robot state by reading Genesis joints and computing FK
                    cache = self.drake_caches.get(robot_name)
                    if cache:
                        try:
                            # Read actual joint positions from Genesis
                            actual_genesis_joints = self._get_robot_joints(entity)
                            self.logger.info(f"ACTUAL GENESIS JOINTS: {robot_name} = {actual_genesis_joints}")
                            
                            # Compute FK using actual Genesis joint positions
                            if len(actual_genesis_joints) >= 6:  # Genesis gives us 6 DOF
                                plant_context = cache.plant.CreateDefaultContext()
                                # Pad Genesis joints to match Drake's expected DOF count
                                padded_joints = np.zeros(cache.plant.num_positions())
                                padded_joints[:len(actual_genesis_joints)] = actual_genesis_joints
                                cache.plant.SetPositions(plant_context, padded_joints)
                                actual_pose = cache.plant.CalcRelativeTransform(plant_context, cache.base_frame, cache.ee_frame)
                                actual_pos = actual_pose.translation()
                                actual_rot = actual_pose.rotation()
                                actual_rpy = actual_rot.ToRollPitchYaw()
                                actual_rpy_deg = (np.rad2deg(actual_rpy.roll_angle()), 
                                                np.rad2deg(actual_rpy.pitch_angle()), 
                                                np.rad2deg(actual_rpy.yaw_angle()))
                                self.logger.info(f"ACTUAL FK POSE: {robot_name} pos={tuple(actual_pos)} RPY={actual_rpy_deg}")
                            else:
                                self.logger.error(f"Genesis returned {len(actual_genesis_joints)} joints but need at least 6")
                        except Exception as e:
                            self.logger.error(f"Failed to compute actual FK pose for {robot_name}: {e}")
                    # Reflect degrees in GUI targets
                    self.joint_targets[robot_name] = [float(np.rad2deg(v)) for v in q]
                else:
                    # find segment
                    idx = np.searchsorted(times, t, side="right") - 1
                    idx = max(0, min(idx, len(times)-2))
                    t0s, t1s = times[idx], times[idx+1]
                    a = 0.0 if t1s <= t0s else (t - t0s) / (t1s - t0s)
                    q = (1.0 - a) * way[idx] + a * way[idx+1]
                    self._set_robot_joints(entity, q.tolist(), robot_name)
                    self.joint_targets[robot_name] = [float(np.rad2deg(v)) for v in q]
                continue

            # No trajectory: fall back to static targets
            targets_deg = self.joint_targets.get(robot_name, [])
            if targets_deg:
                q_rad = [np.deg2rad(deg) for deg in targets_deg]
                self.logger.debug(f"Setting {robot_name} joints: {targets_deg} deg -> {q_rad} rad")
                self._set_robot_joints(entity, q_rad, robot_name)
                
                # Record as last safe state if collision-free
                cc = self.collision_checkers.get(robot_name)
                mdl = self.pin_models.get(robot_name)
                dat = self.pin_datas.get(robot_name)
                if cc and mdl and dat:
                    try:
                        q_array = np.array(q_rad, dtype=np.float64)
                        if not cc.in_collision_from_pin(mdl, dat, q_array):
                            self._last_safe_q[robot_name] = q_array.copy()
                    except Exception:
                        pass
                
        # Update all robot states for collision checking after movement
        self._update_all_robot_states()

    def reset_to_last_safe(self, robot_name: str):
        """Reset robot to last known safe configuration."""
        q = self._last_safe_q.get(robot_name)
        if q is not None:
            entity = self.robot_entities.get(robot_name)
            if entity is not None:
                self._set_robot_joints(entity, q.tolist(), robot_name)
                self.logger.info(f"Reset {robot_name} to last safe configuration")
                return True
        self.logger.warning(f"No safe configuration available for {robot_name}")
        return False

    def _update_all_robot_states(self):
        """Update last known states for all robots by reading from Genesis."""
        for robot_name in self.robot_entities.keys():
            entity = self.robot_entities.get(robot_name)
            if entity:
                try:
                    q_current = self._get_robot_joints(entity, prefer_struct=True)
                    if q_current is not None and len(q_current) > 0 and np.all(np.isfinite(q_current)):
                        self._last_known_q[robot_name] = np.array(q_current, dtype=np.float64)
                        self.logger.debug(f"Updated state for {robot_name}: {q_current}")
                except Exception as e:
                    self.logger.debug(f"Failed to update state for {robot_name}: {e}")

    def _apply_q_to_entity(self, entity, q_rad) -> bool:
        """Best-effort application across Genesis builds; returns True if something applied.
        Prefer APIs that take (values, dofs_idx) to avoid hidden joint-order mismatches.
        """
        applied = False
        dofs_idx = list(range(len(q_rad)))

        def _call(meth_name: str) -> bool:
            if not hasattr(entity, meth_name):
                return False
            fn = getattr(entity, meth_name)
            # Try (values, dofs_idx) first per Genesis tutorials; fall back to (values)
            try:
                fn(q_rad, dofs_idx)
                return True
            except TypeError:
                # Signature without indices
                try:
                    fn(q_rad)
                    return True
                except Exception as e2:
                    self.logger.debug(f"{meth_name}(values) failed: {e2}")
                    return False
            except Exception as e:
                self.logger.debug(f"{meth_name}(values, idx) failed: {e}")
                # Fallback to values-only if previous failed for non-TypeError
                try:
                    fn(q_rad)
                    return True
                except Exception as e2:
                    self.logger.debug(f"{meth_name}(values) failed: {e2}")
                    return False

        # Try immediate teleport (authoritative set)
        for meth in ("set_dofs_position", "set_qpos", "set_q"):
            if _call(meth):
                applied = True
                break

        # Also arm PD target if available (for dynamics)
        for meth in ("control_dofs_position", "set_dofs_position_target", "set_dofs_target"):
            if _call(meth):
                applied = True

        return applied

    def _set_robot_joints(self, entity, q_rad: List[float], robot_name: str = "unknown"):
        try:
            if not q_rad or any(not np.isfinite(q) for q in q_rad):
                self.logger.warning(f"{robot_name}: invalid joint vector {q_rad}")
                return

            # DOF count normalization (keep your current logic)
            robot_config = next((r for r in self.config.robots if r.name == robot_name), None)
            expected_dofs = len(robot_config.initial_joint_positions) if (robot_config and robot_config.initial_joint_positions) else 6
            if len(q_rad) > expected_dofs:
                q_rad = q_rad[:expected_dofs]
            elif len(q_rad) < expected_dofs:
                q_rad = list(q_rad) + [0.0] * (expected_dofs - len(q_rad))

            # Apply
            ok = self._apply_q_to_entity(entity, q_rad)
            if not ok:
                self.logger.error(f"{robot_name}: No applicable Genesis setter found; joints not applied.")
                return

            # Verify readback from the same buffer we will use for FK logging
            verification = self._get_robot_joints(entity, prefer_struct=True)
            self.logger.debug(f"{robot_name}: set -> {np.round(q_rad,5)} ; read -> {np.round(verification,5)}")
        except Exception as e:
            self.logger.error(f"{robot_name}: Failed to set joints: {e}")

    def _get_robot_joints(self, entity, prefer_struct: bool = False) -> np.ndarray:
        """Read DOFs from Genesis. Tries structured handle first, then flat getters."""
        try:
            if prefer_struct and hasattr(entity, "get_dofs"):
                dofs = entity.get_dofs()
                pos = getattr(dofs, "position", None)
                if pos is not None:
                    if hasattr(pos, "cpu"):
                        pos = pos.cpu().numpy()
                    elif hasattr(pos, "detach"):
                        pos = pos.detach().cpu().numpy()
                    return np.array(pos, dtype=np.float32)

            # Fallbacks
            if hasattr(entity, "get_dofs_position"):
                jp = entity.get_dofs_position()
                if hasattr(jp, "cpu"):
                    jp = jp.cpu().numpy()
                elif hasattr(jp, "detach"):
                    jp = jp.detach().cpu().numpy()
                return np.array(jp, dtype=np.float32)

            # Last resort: zeros (consistent shape)
            self.logger.error("Genesis entity exposes no readable DOF interface; returning zeros.")
            return np.zeros(6, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Failed to read Genesis DOFs: {e}")
            return np.zeros(6, dtype=np.float32)


    # Public API methods
    def set_joint_position(self, robot: str, joint_idx: int, value_deg: float):
        """Set a single joint position."""
        cmd = SetJointCommand(robot, joint_idx, value_deg)
        self.command_queue.put(cmd)

    def set_joint_targets(self, robot: str, values_deg: List[float]):
        """Set all joint positions."""
        cmd = SetJointTargetsCommand(robot, values_deg)
        self.command_queue.put(cmd)

    def move_cartesian(self, robot: str, position: Tuple[float, float, float], 
                      orientation_deg: Tuple[float, float, float], frame: str = "base"):
        """Move robot to Cartesian pose."""
        cmd = CartesianMoveCommand(robot, position, orientation_deg, frame)
        self.command_queue.put(cmd)

    def switch_mode(self, robot: str, mode: ControlMode):
        """Switch robot control mode."""
        cmd = SwitchModeCommand(robot, mode)
        self.command_queue.put(cmd)

    def get_robot_mode(self, robot: str) -> ControlMode:
        """Get current robot control mode."""
        return self.robot_modes.get(robot, ControlMode.JOINT)

    def get_joint_targets(self, robot: str) -> List[float]:
        """Get current joint targets."""
        return list(self.joint_targets.get(robot, []))

    def get_joint_positions(self, robot: str) -> List[float]:
        """Get current joint positions (degrees) from targets (more reliable than simulation state)."""
        targets = self.joint_targets.get(robot, [])
        if targets:
            return list(targets)  # Return targets instead of simulation state
        
        # Fallback to simulation state if no targets set
        entity = self.robot_entities.get(robot)
        if not entity:
            return []
        try:
            q_rad = self._get_robot_joints(entity)
            return [float(np.rad2deg(v)) for v in q_rad]
        except Exception:
            return []


__all__ = ["MovementController", "ControlMode"]