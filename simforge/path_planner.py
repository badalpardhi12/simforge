"""Path planning utilities.

Simple interpolation-based path planning.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np


def plan_joint_path(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    max_vel: float = 1.0,
    max_acc: float = 2.0,
    resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Plan a joint space path with time parameterization.
    
    Args:
        q_start: Starting joint configuration
        q_goal: Goal joint configuration  
        max_vel: Maximum joint velocity (rad/s)
        max_acc: Maximum joint acceleration (rad/s²)
        resolution: Number of waypoints
        
    Returns:
        Tuple of (waypoints, timestamps)
    """
    if q_start.shape != q_goal.shape:
        raise ValueError("Start and goal must have same dimensions")
        
    # Simple linear interpolation
    waypoints = np.linspace(q_start, q_goal, resolution)
    
    # Compute time parameterization
    joint_distances = np.abs(q_goal - q_start)
    max_distance = np.max(joint_distances)
    
    # Simple trapezoidal profile
    if max_distance == 0:
        times = np.zeros(resolution)
    else:
        # Time to reach max velocity
        t_accel = max_vel / max_acc
        # Distance covered during acceleration
        d_accel = 0.5 * max_acc * t_accel * t_accel
        
        if 2 * d_accel >= max_distance:
            # Triangular profile
            total_time = 2 * np.sqrt(max_distance / max_acc)
        else:
            # Trapezoidal profile
            d_constant = max_distance - 2 * d_accel
            t_constant = d_constant / max_vel
            total_time = 2 * t_accel + t_constant
            
        times = np.linspace(0, total_time, resolution)
    
    return waypoints, times


def interpolate_cartesian_path(
    poses: List[Tuple[np.ndarray, np.ndarray]], 
    resolution: int = 20
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Interpolate between Cartesian poses.
    
    Args:
        poses: List of (position, quaternion_wxyz) tuples
        resolution: Waypoints per segment
        
    Returns:
        Interpolated poses
    """
    if len(poses) < 2:
        return poses
        
    result = []
    for i in range(len(poses) - 1):
        pos_start, quat_start = poses[i]
        pos_end, quat_end = poses[i + 1]
        
        for j in range(resolution):
            alpha = j / (resolution - 1) if resolution > 1 else 0
            
            # Linear interpolation for position
            pos_interp = (1 - alpha) * pos_start + alpha * pos_end
            
            # SLERP for quaternion (simplified)
            quat_interp = _slerp_quaternion(quat_start, quat_end, alpha)
            
            result.append((pos_interp, quat_interp))
            
    return result


def _slerp_quaternion(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation for quaternions."""
    # Ensure unit quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.dot(q1, q2)
    
    # If dot product is negative, slerp won't take shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot
        
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # Calculate angle between quaternions
    theta_0 = np.arccos(abs(dot))
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2


__all__ = ["plan_joint_path", "interpolate_cartesian_path"]