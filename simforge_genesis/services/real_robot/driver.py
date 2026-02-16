"""High-level UR robot driver integrating connection and trajectory execution.

This module provides a unified interface for controlling UR robots with support
for both simulation-generated trajectories and direct motion commands.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .connection import URRobotConnection, ConnectionConfig, ConnectionState, RobotState
from .trajectory_executor import (
    URTrajectoryExecutor,
    TrajectoryExecutionConfig,
    Trajectory,
    TrajectoryPoint,
    ExecutionState,
)


@dataclass
class RobotDriverConfig:
    """Configuration for UR robot driver."""
    
    robot_ip: str
    """IP address of the robot."""
    
    robot_name: str = "ur_robot"
    """Human-readable name for the robot."""
    
    robot_model: str = "UR5e"
    """Robot model: UR3e, UR5e, UR10e, UR16e, UR20."""
    
    rtde_frequency: float = 500.0
    """RTDE communication frequency in Hz."""
    
    use_external_control_cap: bool = False
    """Use the ExternalControl URCap."""
    
    # Trajectory execution settings
    max_joint_velocity: float = 1.05
    """Maximum joint velocity in rad/s."""
    
    max_joint_acceleration: float = 1.4
    """Maximum joint acceleration in rad/s²."""
    
    velocity_scaling: float = 1.0
    """Global velocity scaling (0.0 to 1.0)."""
    
    blend_radius: float = 0.03
    """Blend radius for moveJ path execution (rad). Use ~0.03-0.05 for smooth motion."""
    
    settling_time: float = 0.3
    """Time to wait for robot to settle after motion."""
    
    # TCP settings
    tcp_offset: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """TCP offset [x, y, z, rx, ry, rz] in meters and radians."""
    
    payload_mass: float = 0.0
    """Payload mass in kg."""
    
    payload_cog: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Payload center of gravity [x, y, z] in meters."""
    
    # Safety settings
    enable_safety_checks: bool = True
    """Enable safety checks before motion."""


class URRobotDriver:
    """High-level driver for controlling UR robots.
    
    This class provides a unified interface for:
    - Robot connection management
    - Trajectory execution from simulation plans
    - Direct motion commands (moveJ, moveL)
    - State monitoring and feedback
    
    Example:
        >>> config = RobotDriverConfig(
        ...     robot_ip="192.168.1.9",
        ...     robot_name="nakul_ur5e",
        ...     robot_model="UR5e",
        ... )
        >>> driver = URRobotDriver(config)
        >>> driver.connect()
        >>> driver.move_to_joints([0, -90, 0, -90, 0, 0], degrees=True)
        >>> driver.disconnect()
    """
    
    # Joint limits by robot model (in radians)
    JOINT_LIMITS = {
        "UR3e": [(-2*math.pi, 2*math.pi)] * 6,
        "UR5e": [(-2*math.pi, 2*math.pi)] * 6,
        "UR10e": [(-2*math.pi, 2*math.pi)] * 6,
        "UR16e": [(-2*math.pi, 2*math.pi)] * 6,
        "UR20": [(-2*math.pi, 2*math.pi)] * 6,
    }
    
    def __init__(
        self,
        config: RobotDriverConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(f"ur_driver.{config.robot_name}")
        
        # Create connection config
        conn_config = ConnectionConfig(
            robot_ip=config.robot_ip,
            rtde_frequency=config.rtde_frequency,
            use_external_control_cap=config.use_external_control_cap,
        )
        self._connection = URRobotConnection(conn_config, self._logger.getChild("conn"))
        
        # Create trajectory executor config
        exec_config = TrajectoryExecutionConfig(
            servo_frequency=config.rtde_frequency,
            max_joint_velocity=config.max_joint_velocity,
            max_joint_acceleration=config.max_joint_acceleration,
            velocity_scaling=config.velocity_scaling,
            blend_radius=config.blend_radius,
            settling_time=config.settling_time,
        )
        self._executor = URTrajectoryExecutor(
            self._connection,
            exec_config,
            self._logger.getChild("exec"),
        )
        
        self._initialized = False
    
    @property
    def config(self) -> RobotDriverConfig:
        """Driver configuration."""
        return self._config
    
    @property
    def is_connected(self) -> bool:
        """Whether robot is connected."""
        return self._connection.is_connected
    
    @property
    def connection_state(self) -> ConnectionState:
        """Current connection state."""
        return self._connection.state
    
    @property
    def execution_state(self) -> ExecutionState:
        """Current trajectory execution state."""
        return self._executor.state
    
    @property
    def robot_state(self) -> RobotState:
        """Current robot state."""
        return self._connection.get_state()
    
    def connect(self) -> bool:
        """Connect to the robot.
        
        Returns:
            True if connection successful.
        """
        self._logger.info("Connecting to %s at %s", self._config.robot_name, self._config.robot_ip)
        
        success = self._connection.connect()
        if not success:
            return False
        
        # Initialize robot settings
        try:
            self._initialize_robot()
            self._initialized = True
        except Exception as exc:
            self._logger.error("Robot initialization failed: %s", exc)
            return False
        
        return True
    
    def disconnect(self) -> None:
        """Disconnect from the robot."""
        self._logger.info("Disconnecting from %s", self._config.robot_name)
        
        # Stop any active motion
        if self._executor.state == ExecutionState.EXECUTING:
            self._executor.stop()
        
        self._connection.disconnect()
        self._initialized = False
    
    def _initialize_robot(self) -> None:
        """Initialize robot settings (TCP, payload, etc.)."""
        rtde_c = self._connection.control_interface
        if rtde_c is None:
            return
        
        # Set TCP offset
        if self._config.tcp_offset != (0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
            self._logger.info("Setting TCP offset: %s", self._config.tcp_offset)
            rtde_c.setTcp(list(self._config.tcp_offset))
        
        # Set payload
        if self._config.payload_mass > 0:
            self._logger.info(
                "Setting payload: %.2f kg at %s",
                self._config.payload_mass,
                self._config.payload_cog,
            )
            rtde_c.setPayload(
                self._config.payload_mass,
                list(self._config.payload_cog),
            )
    
    def get_joint_positions(self, degrees: bool = False) -> Tuple[float, ...]:
        """Get current joint positions.
        
        Args:
            degrees: If True, return values in degrees.
        
        Returns:
            Tuple of 6 joint positions.
        """
        state = self._connection.get_state()
        positions = state.actual_q
        
        if degrees:
            return tuple(math.degrees(q) for q in positions)
        return positions
    
    def get_tcp_pose(self) -> Tuple[float, ...]:
        """Get current TCP pose.
        
        Returns:
            Tuple [x, y, z, rx, ry, rz] in meters and radians.
        """
        state = self._connection.get_state()
        return state.actual_tcp_pose
    
    def is_ready(self) -> bool:
        """Check if robot is ready for motion.
        
        Returns:
            True if robot is connected, not in error, and ready to move.
        """
        if not self.is_connected:
            return False
        
        state = self._connection.get_state()
        
        if state.emergency_stopped:
            self._logger.warning("Robot is in emergency stop")
            return False
        
        if state.protective_stopped:
            self._logger.warning("Robot is in protective stop")
            return False
        
        # Robot modes: 7 = RUNNING (normal mode)
        if state.robot_mode < 5:
            self._logger.warning("Robot not in running mode: %d", state.robot_mode)
            return False
        
        return True
    
    def move_to_joints(
        self,
        target_joints: Sequence[float],
        *,
        degrees: bool = False,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """Move robot to target joint positions.
        
        Args:
            target_joints: Target joint positions (6 values).
            degrees: If True, input is in degrees.
            velocity: Joint velocity in rad/s (or deg/s if degrees=True).
            acceleration: Joint acceleration in rad/s² (or deg/s² if degrees=True).
            blocking: Wait for motion to complete.
        
        Returns:
            True if motion completed successfully.
        """
        if not self.is_ready():
            self._logger.error("Robot not ready for motion")
            return False
        
        # Convert to radians if needed
        if degrees:
            target = [math.radians(q) for q in target_joints]
        else:
            target = list(target_joints)
        
        # Apply defaults
        vel = velocity or self._config.max_joint_velocity
        acc = acceleration or self._config.max_joint_acceleration
        
        if degrees and velocity:
            vel = math.radians(vel)
        if degrees and acceleration:
            acc = math.radians(acc)
        
        # Scale by velocity factor
        vel *= self._config.velocity_scaling
        
        # Safety check
        if self._config.enable_safety_checks:
            if not self._check_joint_limits(target):
                return False
        
        self._logger.info(
            "Moving to joints: %s (vel=%.2f, acc=%.2f)",
            [f"{math.degrees(q):.1f}°" for q in target],
            vel,
            acc,
        )
        
        rtde_c = self._connection.control_interface
        if rtde_c is None:
            return False
        
        try:
            if blocking:
                return rtde_c.moveJ(target, vel, acc, False)
            else:
                return rtde_c.moveJ(target, vel, acc, True)
        except Exception as exc:
            self._logger.error("moveJ failed: %s", exc)
            return False
    
    def move_to_pose(
        self,
        target_pose: Sequence[float],
        *,
        velocity: float = 0.25,
        acceleration: float = 0.5,
        blocking: bool = True,
    ) -> bool:
        """Move robot to target TCP pose (Cartesian).
        
        Args:
            target_pose: Target pose [x, y, z, rx, ry, rz] in meters and radians.
            velocity: TCP velocity in m/s.
            acceleration: TCP acceleration in m/s².
            blocking: Wait for motion to complete.
        
        Returns:
            True if motion completed successfully.
        """
        if not self.is_ready():
            return False
        
        rtde_c = self._connection.control_interface
        if rtde_c is None:
            return False
        
        self._logger.info(
            "Moving to pose: [%.3f, %.3f, %.3f, %.2f, %.2f, %.2f]",
            *target_pose,
        )
        
        try:
            if blocking:
                return rtde_c.moveL(list(target_pose), velocity, acceleration, False)
            else:
                return rtde_c.moveL(list(target_pose), velocity, acceleration, True)
        except Exception as exc:
            self._logger.error("moveL failed: %s", exc)
            return False
    
    def execute_trajectory(
        self,
        trajectory: Trajectory,
        *,
        blocking: bool = True,
        use_servo: bool = True,
    ) -> bool:
        """Execute a pre-planned trajectory.
        
        Args:
            trajectory: Trajectory object with waypoints and timing.
            blocking: Wait for completion.
            use_servo: Use servoJ for precise timing (vs moveJ path).
        
        Returns:
            True if trajectory executed successfully.
        """
        if not self.is_ready():
            return False
        
        self._logger.info(
            "Executing trajectory: %d points, %.2fs duration",
            len(trajectory.points),
            trajectory.total_duration,
        )
        
        return self._executor.execute(
            trajectory,
            blocking=blocking,
            use_servo=use_servo,
        )
    
    def execute_plan_pose(
        self,
        pose_data: Dict[str, Any],
        *,
        blocking: bool = True,
        velocity_scale: float = 1.0,
    ) -> bool:
        """Execute a single pose from a simforge plan.
        
        Args:
            pose_data: Dict with 'waypoints' key containing joint trajectories.
            blocking: Wait for completion.
            velocity_scale: Scale trajectory speed.
        
        Returns:
            True if execution successful.
        """
        if "waypoints" not in pose_data:
            self._logger.error("Pose data missing 'waypoints' key")
            return False
        
        waypoints = pose_data["waypoints"]
        times_s = pose_data.get("times_s")
        duration = pose_data.get("duration")
        
        # Scale duration by velocity
        if duration and velocity_scale != 1.0:
            duration = duration / velocity_scale
        
        trajectory = Trajectory.from_waypoints(waypoints, times_s, duration)
        
        return self.execute_trajectory(trajectory, blocking=blocking)
    
    def execute_plan_file(
        self,
        plan_path: Union[str, Path],
        *,
        object_key: Optional[str] = None,
        pose_keys: Optional[List[str]] = None,
        blocking: bool = True,
        velocity_scale: float = 1.0,
        inter_pose_delay: float = 0.5,
    ) -> Dict[str, bool]:
        """Execute poses from a simforge plan JSON file.
        
        Args:
            plan_path: Path to plan JSON file.
            object_key: Specific object to execute (e.g., 'face_link').
            pose_keys: Specific poses to execute (e.g., ['pose_1', 'pose_2']).
            blocking: Wait for each pose to complete.
            velocity_scale: Scale trajectory speed.
            inter_pose_delay: Delay between poses in seconds.
        
        Returns:
            Dict mapping pose keys to success status.
        """
        plan_path = Path(plan_path)
        if not plan_path.exists():
            self._logger.error("Plan file not found: %s", plan_path)
            return {}
        
        with open(plan_path) as f:
            plan_data = json.load(f)
        
        results: Dict[str, bool] = {}
        
        # Get object data
        if object_key:
            if object_key not in plan_data:
                self._logger.error("Object '%s' not found in plan", object_key)
                return {}
            object_data = {object_key: plan_data[object_key]}
        else:
            object_data = plan_data
        
        # Execute each pose
        for obj_name, obj_poses in object_data.items():
            self._logger.info("Executing object: %s", obj_name)
            
            for pose_key, pose_value in obj_poses.items():
                if pose_key == "pose_in_robot_frame":
                    continue
                
                if pose_keys and pose_key not in pose_keys:
                    continue
                
                if not isinstance(pose_value, dict) or "waypoints" not in pose_value:
                    continue
                
                self._logger.info("Executing %s/%s", obj_name, pose_key)
                
                success = self.execute_plan_pose(
                    pose_value,
                    blocking=blocking,
                    velocity_scale=velocity_scale,
                )
                
                results[f"{obj_name}/{pose_key}"] = success
                
                if not success:
                    self._logger.warning("Pose %s failed", pose_key)
                    if blocking:
                        break
                
                if inter_pose_delay > 0:
                    time.sleep(inter_pose_delay)
        
        return results
    
    def stop(self, deceleration: float = 2.0) -> None:
        """Stop all robot motion.
        
        Args:
            deceleration: Deceleration rate.
        """
        self._logger.info("Stopping robot motion")
        
        self._executor.stop(deceleration)
        
        rtde_c = self._connection.control_interface
        if rtde_c:
            try:
                rtde_c.stopJ(deceleration)
            except Exception:
                pass
    
    def _check_joint_limits(self, joints: Sequence[float]) -> bool:
        """Check if joint positions are within limits.
        
        Args:
            joints: Joint positions in radians.
        
        Returns:
            True if all joints within limits.
        """
        limits = self.JOINT_LIMITS.get(self._config.robot_model, self.JOINT_LIMITS["UR5e"])
        
        for i, (q, (q_min, q_max)) in enumerate(zip(joints, limits)):
            if q < q_min or q > q_max:
                self._logger.error(
                    "Joint %d out of limits: %.2f not in [%.2f, %.2f]",
                    i + 1,
                    math.degrees(q),
                    math.degrees(q_min),
                    math.degrees(q_max),
                )
                return False
        
        return True
    
    def __enter__(self) -> "URRobotDriver":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


__all__ = ["URRobotDriver", "RobotDriverConfig"]
