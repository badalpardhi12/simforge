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

from .config_reader import SimforgeConfig
from .genesis_renderer import GenesisRenderer
from .ik_solver import solve_ik, HAS_PINOCCHIO
from .path_planner import plan_joint_path
from .collision_checker import CollisionChecker
from .logging_utils import setup_logging


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
        
        # Pinocchio models for IK
        self.pin_models: Dict[str, Any] = {}
        self.pin_datas: Dict[str, Any] = {}
        
        # Threading
        self.command_queue: queue.Queue[Command] = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Genesis scene
        self.scene = None
        
        self._initialize_robots()

    def _initialize_robots(self):
        """Initialize robot state and models."""
        for robot_config in self.config.robots:
            name = robot_config.name
            self.robot_modes[name] = ControlMode.JOINT
            self.joint_targets[name] = list(robot_config.initial_joint_positions or [0.0] * 6)
            
            # Initialize collision checker
            try:
                self.collision_checkers[name] = CollisionChecker(
                    robot_config.urdf, self.logger
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize collision checker for {name}: {e}"
                )
            
            # Initialize Pinocchio model for IK
            if HAS_PINOCCHIO:
                try:
                    import pinocchio as pin
                    model = pin.buildModelFromUrdf(robot_config.urdf)
                    data = model.createData()
                    self.pin_models[name] = model
                    self.pin_datas[name] = data
                    self.logger.info(f"Loaded Pinocchio model for {name}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load Pinocchio model for {name}: {e}"
                    )

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
                self._set_robot_joints(entity, q_rad)
        
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
        """Execute a Cartesian movement command."""
        robot_name = cmd.robot
        
        if robot_name not in self.pin_models:
            self.logger.error(f"No IK model for robot {robot_name}")
            return
            
        # Get robot configuration
        robot_config = next((r for r in self.config.robots if r.name == robot_name), None)
        if not robot_config or not robot_config.end_effector_link:
            self.logger.error(f"No end effector link configured for {robot_name}")
            return
        
        # Current joint state
        entity = self.robot_entities[robot_name]
        q_current = self._get_robot_joints(entity)
        
        # Convert RPY to quaternion
        roll, pitch, yaw = [np.deg2rad(x) for x in cmd.orientation_deg]
        quat = self._rpy_to_quaternion(roll, pitch, yaw)
        
        # Solve IK
        model = self.pin_models[robot_name]
        data = self.pin_datas[robot_name]
        
        q_solution = solve_ik(
            model, data,
            robot_config.end_effector_link,
            q_current,
            cmd.position,
            quat,
            max_iters=100
        )
        
        if q_solution is None:
            self.logger.warning(f"IK failed for {robot_name}")
            return
        
        # Plan path
        waypoints, times = plan_joint_path(q_current, q_solution)
        
        # Execute path (simplified - just set final position)
        q_deg = [float(np.rad2deg(q)) for q in q_solution]
        self.joint_targets[robot_name] = q_deg
        
        self.logger.info(f"Cartesian move executed for {robot_name}")

    def _update_robots(self):
        """Update robot joint positions."""
        for robot_name, targets_deg in self.joint_targets.items():
            entity = self.robot_entities.get(robot_name)
            if entity and targets_deg:
                q_rad = [np.deg2rad(deg) for deg in targets_deg]
                self._set_robot_joints(entity, q_rad)

    def _set_robot_joints(self, entity, q_rad: List[float]):
        """Set robot joint positions."""
        try:
            entity.set_dofs_position(q_rad)
        except Exception as e:
            self.logger.debug(f"Failed to set joint positions: {e}")

    def _get_robot_joints(self, entity) -> np.ndarray:
        """Get current robot joint positions."""
        try:
            joints = entity.get_joints()
            return np.array([joint.qpos for joint in joints], dtype=np.float32)
        except Exception:
            return np.zeros(6, dtype=np.float32)

    def _rpy_to_quaternion(
        self, roll: float, pitch: float, yaw: float
    ) -> Tuple[float, float, float, float]:
        """Convert roll-pitch-yaw to quaternion (w, x, y, z)."""
        # Pre-compute half angles
        half_yaw = yaw * 0.5
        half_pitch = pitch * 0.5
        half_roll = roll * 0.5
        
        cy = np.cos(half_yaw)
        sy = np.sin(half_yaw)
        cp = np.cos(half_pitch)
        sp = np.sin(half_pitch)
        cr = np.cos(half_roll)
        sr = np.sin(half_roll)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return (w, x, y, z)

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