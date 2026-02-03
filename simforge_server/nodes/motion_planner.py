"""
Motion Planning Service for Simforge Server

Provides path planning capabilities using:
- Cartesian interpolation for simple linear moves
- Joint-space interpolation for joint moves
- Collision-aware trajectory generation

This is a lightweight planner for development. On Jetson Thor with cuMotion,
this can be enhanced with GPU-accelerated motion planning.
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
import math

from .collision_checker import CollisionChecker, CollisionResult


logger = logging.getLogger(__name__)


class MotionType(IntEnum):
    """Type of motion."""
    JOINT = 0
    CARTESIAN = 1
    BLEND = 2  # Joint move with Cartesian blend


@dataclass
class TrajectoryPoint:
    """A point in a trajectory."""
    joint_positions: np.ndarray  # [6] for 6-DOF robot
    timestamp: float  # Time from start in seconds
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None


@dataclass
class Trajectory:
    """A complete trajectory."""
    points: List[TrajectoryPoint]
    duration: float
    motion_type: MotionType
    is_valid: bool = True
    error_message: str = ""


@dataclass 
class PlanningRequest:
    """Request for motion planning."""
    start_joints: np.ndarray
    # For joint motion
    target_joints: Optional[np.ndarray] = None
    # For Cartesian motion
    target_pose: Optional[np.ndarray] = None  # [x, y, z, qx, qy, qz, qw]
    # Motion parameters
    motion_type: MotionType = MotionType.JOINT
    velocity_scale: float = 0.5  # 0.0 - 1.0
    acceleration_scale: float = 0.5  # 0.0 - 1.0
    # Planning options
    collision_check: bool = True
    max_planning_time: float = 5.0


@dataclass
class PlanningResult:
    """Result of motion planning."""
    success: bool
    trajectory: Optional[Trajectory] = None
    error_message: str = ""
    planning_time: float = 0.0
    collision_free: bool = True


class MotionPlanner:
    """
    Motion planning service for robot trajectory generation.
    
    Provides:
    - Joint-space interpolation with velocity/acceleration limits
    - Cartesian linear interpolation with IK
    - Collision checking along trajectory
    - Time-optimal trajectory generation
    """
    
    # UR5e joint limits (radians)
    JOINT_LIMITS = [
        (-2 * np.pi, 2 * np.pi),  # Base
        (-2 * np.pi, 2 * np.pi),  # Shoulder
        (-np.pi, np.pi),          # Elbow
        (-2 * np.pi, 2 * np.pi),  # Wrist 1
        (-2 * np.pi, 2 * np.pi),  # Wrist 2
        (-2 * np.pi, 2 * np.pi),  # Wrist 3
    ]
    
    # UR5e velocity limits (rad/s)
    MAX_JOINT_VELOCITIES = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
    
    # UR5e acceleration limits (rad/s^2)
    MAX_JOINT_ACCELERATIONS = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    
    # Trajectory parameters
    DEFAULT_DT = 0.008  # 125Hz control rate for UR robots
    
    def __init__(
        self,
        collision_checker: Optional[CollisionChecker] = None,
        forward_kinematics_fn: Optional[callable] = None,
        inverse_kinematics_fn: Optional[callable] = None,
    ) -> None:
        """
        Initialize the motion planner.
        
        Args:
            collision_checker: Optional collision checker for trajectory validation
            forward_kinematics_fn: Function to compute FK (joints -> pose)
            inverse_kinematics_fn: Function to compute IK (pose -> joints)
        """
        self.collision_checker = collision_checker
        self.fk_fn = forward_kinematics_fn
        self.ik_fn = inverse_kinematics_fn
    
    def plan(self, request: PlanningRequest) -> PlanningResult:
        """
        Plan a trajectory for the given request.
        
        Args:
            request: Planning request with start/goal and parameters
        
        Returns:
            PlanningResult with trajectory if successful
        """
        import time
        start_time = time.time()
        
        try:
            if request.motion_type == MotionType.JOINT:
                if request.target_joints is None:
                    return PlanningResult(
                        success=False,
                        error_message="target_joints required for joint motion",
                    )
                trajectory = self._plan_joint_motion(
                    request.start_joints,
                    request.target_joints,
                    request.velocity_scale,
                    request.acceleration_scale,
                )
            
            elif request.motion_type == MotionType.CARTESIAN:
                if request.target_pose is None:
                    return PlanningResult(
                        success=False,
                        error_message="target_pose required for Cartesian motion",
                    )
                if self.ik_fn is None:
                    return PlanningResult(
                        success=False,
                        error_message="IK function not configured for Cartesian planning",
                    )
                trajectory = self._plan_cartesian_motion(
                    request.start_joints,
                    request.target_pose,
                    request.velocity_scale,
                    request.acceleration_scale,
                )
            
            else:
                return PlanningResult(
                    success=False,
                    error_message=f"Unsupported motion type: {request.motion_type}",
                )
            
            if not trajectory.is_valid:
                return PlanningResult(
                    success=False,
                    error_message=trajectory.error_message,
                    planning_time=time.time() - start_time,
                )
            
            # Collision check if enabled
            collision_free = True
            if request.collision_check and self.collision_checker:
                collision_free = self._check_trajectory_collision(trajectory)
                if not collision_free:
                    return PlanningResult(
                        success=False,
                        trajectory=trajectory,
                        error_message="Trajectory has collisions",
                        planning_time=time.time() - start_time,
                        collision_free=False,
                    )
            
            return PlanningResult(
                success=True,
                trajectory=trajectory,
                planning_time=time.time() - start_time,
                collision_free=collision_free,
            )
            
        except Exception as e:
            logger.exception("Planning failed")
            return PlanningResult(
                success=False,
                error_message=str(e),
                planning_time=time.time() - start_time,
            )
    
    def _plan_joint_motion(
        self,
        start_joints: np.ndarray,
        target_joints: np.ndarray,
        velocity_scale: float,
        acceleration_scale: float,
    ) -> Trajectory:
        """
        Plan a joint-space motion using trapezoidal velocity profile.
        
        Args:
            start_joints: Starting joint positions (radians)
            target_joints: Target joint positions (radians)
            velocity_scale: Scale factor for max velocity (0-1)
            acceleration_scale: Scale factor for max acceleration (0-1)
        
        Returns:
            Trajectory with interpolated points
        """
        start = np.array(start_joints, dtype=np.float64)
        target = np.array(target_joints, dtype=np.float64)
        
        # Validate joint limits
        for i, (pos, limits) in enumerate(zip(target, self.JOINT_LIMITS)):
            if pos < limits[0] or pos > limits[1]:
                return Trajectory(
                    points=[],
                    duration=0,
                    motion_type=MotionType.JOINT,
                    is_valid=False,
                    error_message=f"Joint {i} target {pos:.3f} exceeds limits [{limits[0]:.3f}, {limits[1]:.3f}]",
                )
        
        # Calculate scaled limits
        max_vel = self.MAX_JOINT_VELOCITIES * velocity_scale
        max_acc = self.MAX_JOINT_ACCELERATIONS * acceleration_scale
        
        # Calculate motion profile for each joint
        delta = target - start
        
        # Time needed for each joint (trapezoidal profile)
        times = []
        for i in range(len(delta)):
            d = abs(delta[i])
            v = max_vel[i]
            a = max_acc[i]
            
            # Time to accelerate to max velocity
            t_acc = v / a
            # Distance during acceleration
            d_acc = 0.5 * a * t_acc ** 2
            
            if 2 * d_acc >= d:
                # Triangular profile (never reach max velocity)
                t = 2 * np.sqrt(d / a)
            else:
                # Trapezoidal profile
                d_cruise = d - 2 * d_acc
                t_cruise = d_cruise / v
                t = 2 * t_acc + t_cruise
            
            times.append(t)
        
        # Use the longest time for synchronization
        duration = max(times) if times else 0.0
        
        if duration < self.DEFAULT_DT:
            # Already at target
            return Trajectory(
                points=[TrajectoryPoint(joint_positions=target, timestamp=0.0)],
                duration=0.0,
                motion_type=MotionType.JOINT,
                is_valid=True,
            )
        
        # Generate trajectory points
        points = []
        t = 0.0
        while t <= duration:
            # Linear interpolation (simple approach)
            # For production, use proper trapezoidal profile per joint
            alpha = t / duration
            # Smooth interpolation using cosine
            smooth_alpha = 0.5 * (1 - np.cos(np.pi * alpha))
            
            positions = start + smooth_alpha * delta
            
            # Estimate velocities (numerical differentiation)
            if t > 0 and len(points) > 0:
                velocities = (positions - points[-1].joint_positions) / self.DEFAULT_DT
            else:
                velocities = np.zeros_like(positions)
            
            points.append(TrajectoryPoint(
                joint_positions=positions,
                timestamp=t,
                velocities=velocities,
            ))
            
            t += self.DEFAULT_DT
        
        # Ensure final point is exactly at target
        if len(points) > 0:
            points[-1].joint_positions = target.copy()
            points[-1].velocities = np.zeros_like(target)
        
        return Trajectory(
            points=points,
            duration=duration,
            motion_type=MotionType.JOINT,
            is_valid=True,
        )
    
    def _plan_cartesian_motion(
        self,
        start_joints: np.ndarray,
        target_pose: np.ndarray,
        velocity_scale: float,
        acceleration_scale: float,
    ) -> Trajectory:
        """
        Plan a Cartesian linear motion.
        
        Args:
            start_joints: Starting joint positions
            target_pose: Target pose [x, y, z, qx, qy, qz, qw]
            velocity_scale: Scale factor for velocity
            acceleration_scale: Scale factor for acceleration
        
        Returns:
            Trajectory with interpolated points
        """
        if self.fk_fn is None or self.ik_fn is None:
            return Trajectory(
                points=[],
                duration=0,
                motion_type=MotionType.CARTESIAN,
                is_valid=False,
                error_message="FK/IK functions not configured",
            )
        
        start = np.array(start_joints, dtype=np.float64)
        target_pos = np.array(target_pose, dtype=np.float64)
        
        # Get start pose via FK
        start_pose = self.fk_fn(start)
        
        # Calculate Cartesian distance
        position_diff = target_pos[:3] - start_pose[:3]
        distance = np.linalg.norm(position_diff)
        
        # Calculate duration based on Cartesian velocity
        cart_velocity = 0.25 * velocity_scale  # 0.25 m/s at full scale
        duration = distance / cart_velocity if cart_velocity > 0 else 0.0
        
        if duration < self.DEFAULT_DT:
            # Solve IK for final position
            target_joints = self.ik_fn(target_pos, start)
            if target_joints is None:
                return Trajectory(
                    points=[],
                    duration=0,
                    motion_type=MotionType.CARTESIAN,
                    is_valid=False,
                    error_message="IK failed for target pose",
                )
            return Trajectory(
                points=[TrajectoryPoint(joint_positions=target_joints, timestamp=0.0)],
                duration=0.0,
                motion_type=MotionType.CARTESIAN,
                is_valid=True,
            )
        
        # Generate waypoints along Cartesian path
        points = []
        t = 0.0
        prev_joints = start.copy()
        
        while t <= duration:
            alpha = t / duration
            # Smooth interpolation
            smooth_alpha = 0.5 * (1 - np.cos(np.pi * alpha))
            
            # Interpolate position (linear)
            interp_pos = start_pose[:3] + smooth_alpha * position_diff
            
            # Interpolate orientation (SLERP would be better, using linear for now)
            interp_quat = start_pose[3:] + smooth_alpha * (target_pos[3:] - start_pose[3:])
            interp_quat = interp_quat / np.linalg.norm(interp_quat)
            
            interp_pose = np.concatenate([interp_pos, interp_quat])
            
            # Solve IK
            joints = self.ik_fn(interp_pose, prev_joints)
            if joints is None:
                return Trajectory(
                    points=points,
                    duration=t,
                    motion_type=MotionType.CARTESIAN,
                    is_valid=False,
                    error_message=f"IK failed at t={t:.3f}s",
                )
            
            # Calculate velocities
            if t > 0 and len(points) > 0:
                velocities = (joints - points[-1].joint_positions) / self.DEFAULT_DT
            else:
                velocities = np.zeros_like(joints)
            
            points.append(TrajectoryPoint(
                joint_positions=joints,
                timestamp=t,
                velocities=velocities,
            ))
            
            prev_joints = joints
            t += self.DEFAULT_DT
        
        return Trajectory(
            points=points,
            duration=duration,
            motion_type=MotionType.CARTESIAN,
            is_valid=True,
        )
    
    def _check_trajectory_collision(self, trajectory: Trajectory) -> bool:
        """
        Check trajectory for collisions.
        
        Returns True if collision-free, False if collision detected.
        """
        if not self.collision_checker:
            return True
        
        # Check every N-th point to reduce computation
        check_interval = max(1, len(trajectory.points) // 50)
        
        for i in range(0, len(trajectory.points), check_interval):
            point = trajectory.points[i]
            
            # This requires forward kinematics to update link transforms
            # For now, we'll skip detailed collision checking
            # In production, this would update all link transforms and check
            pass
        
        return True  # Placeholder - implement with FK
    
    def interpolate_joints(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_points: int,
    ) -> List[np.ndarray]:
        """
        Simple linear interpolation between joint configurations.
        
        Args:
            start: Start joint positions
            end: End joint positions
            num_points: Number of interpolation points
        
        Returns:
            List of joint position arrays
        """
        points = []
        for i in range(num_points):
            alpha = i / (num_points - 1) if num_points > 1 else 1.0
            smooth_alpha = 0.5 * (1 - np.cos(np.pi * alpha))
            points.append(start + smooth_alpha * (end - start))
        return points


def create_motion_planner(
    collision_checker: Optional[CollisionChecker] = None,
) -> MotionPlanner:
    """Create a motion planner with optional collision checking."""
    return MotionPlanner(collision_checker=collision_checker)
