"""
Spherical pose generation for proto-sim workflow.

This module contains the geometry helpers and pose sampling logic ported from
simforge_new/control/proto_simulation.py. It generates camera/tool poses in
spherical coordinates relative to a target object's reference frame.

The pose generation follows the SpatialScout sampling pipeline:
1. Define pivot point at (horiz, 0, vert) in target frame
2. Compute position using spherical coordinates (pitch, yaw, distance) from pivot
3. Compute orientation using look-at constraint (point Z-axis at pivot with roll)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProtoSimParameters:
    """Fully expanded parameter sequences for pose sampling."""
    horiz: Sequence[float]      # Horizontal offset in mm
    vert: Sequence[float]       # Vertical offset in mm  
    distance: Sequence[float]   # Distance from pivot in mm
    roll: Sequence[float]       # Roll around look-at axis in degrees
    pitch: Sequence[float]      # Pitch (elevation) angle in degrees
    yaw: Sequence[float]        # Yaw (azimuth) angle in degrees


@dataclass(frozen=True)
class ProtoPose:
    """Pose expressed in the target object's reference frame."""
    parameters: Dict[str, float]
    position_m: Tuple[float, float, float]
    orientation_deg: Tuple[float, float, float]  # RPY
    orientation_quat_xyzw: Tuple[float, float, float, float]
    
    def get_name(self) -> str:
        """Generate a descriptive name for this pose."""
        p = self.parameters
        return f"H{p['horiz']:.0f}_V{p['vert']:.0f}_D{p['distance']:.0f}_R{p['roll']:.0f}_P{p['pitch']:.0f}_Y{p['yaw']:.0f}"


@dataclass
class WorldPose:
    """Pose expressed in robot's base_link frame (world coordinates)."""
    name: str
    position_m: Tuple[float, float, float]  # x, y, z in base_link frame
    orientation_quat_xyzw: Tuple[float, float, float, float]  # quaternion in base_link frame
    parameters: Dict[str, float]  # Original sampling parameters

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "position": list(self.position_m),
            "orientation": list(self.orientation_quat_xyzw),
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# Geometry helpers (from SpatialScout reference implementation)
# ---------------------------------------------------------------------------

def _quat_from_matrix(R: List[List[float]]) -> Tuple[float, float, float, float]:
    """Compute quaternion (x, y, z, w) from a 3x3 rotation matrix."""
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


def _normalize_quaternion(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Normalize a quaternion (x, y, z, w)."""
    x, y, z, w = quat
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/norm, y/norm, z/norm, w/norm)


def _quat_to_rpy_deg(quat_xyzw: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """Convert quaternion (x,y,z,w) to roll-pitch-yaw in degrees."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _rpy_to_quat_xyzw(roll_rad: float, pitch_rad: float, yaw_rad: float) -> Tuple[float, float, float, float]:
    """Convert roll-pitch-yaw (radians) to quaternion (x,y,z,w)."""
    cr, sr = math.cos(roll_rad * 0.5), math.sin(roll_rad * 0.5)
    cp, sp = math.cos(pitch_rad * 0.5), math.sin(pitch_rad * 0.5)
    cy, sy = math.cos(yaw_rad * 0.5), math.sin(yaw_rad * 0.5)
    
    return (
        sr * cp * cy + cr * sp * sy,  # x
        cr * sp * cy - sr * cp * sy,  # y
        cr * cp * sy + sr * sp * cy,  # z
        cr * cp * cy - sr * sp * sy,  # w
    )


def _quat_multiply(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Multiply two quaternions (x,y,z,w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _mm_to_m(value: float) -> float:
    """Convert millimeters to meters."""
    return value / 1000.0


def _spherical_to_cartesian(pitch_deg: float, yaw_deg: float, distance_mm: float) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates to Cartesian offset.
    
    Args:
        pitch_deg: Elevation angle in degrees (positive = up)
        yaw_deg: Azimuth angle in degrees (0 = +Y axis, positive = towards +X)
        distance_mm: Radial distance in millimeters
        
    Returns:
        (x, y, z) offset in meters
    """
    d = _mm_to_m(distance_mm)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    
    return (
        sy * d * cp,   # X: lateral offset
        cy * d * cp,   # Y: forward offset  
        d * sp,        # Z: vertical offset
    )


def _look_at_quaternion(
    position: Tuple[float, float, float],
    target: Tuple[float, float, float],
    roll_deg: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Compute quaternion that points Z-axis from position towards target with optional roll.
    
    Args:
        position: Source position (x, y, z)
        target: Target position to look at (x, y, z)
        roll_deg: Roll angle around look-at axis in degrees
        
    Returns:
        Quaternion (x, y, z, w)
    """
    src = np.asarray(position, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    
    # Z-axis points from position to target
    z_axis = tgt - src
    norm = np.linalg.norm(z_axis)
    if norm < 1e-9:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_axis = z_axis / norm
    
    # Choose up vector (prefer world Z, fall back to Y if looking straight up/down)
    up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(up_axis, z_axis)) >= 0.95:
        up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    
    # X-axis is perpendicular to Z and up
    x_axis = np.cross(up_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm
    
    # Y-axis completes the right-handed frame
    y_axis = np.cross(z_axis, x_axis)
    
    # Build rotation matrix [X|Y|Z] and convert to quaternion
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    quat_xyzw = _quat_from_matrix(rot_matrix.tolist())
    
    # Apply roll around Z-axis
    if abs(roll_deg) > 1e-6:
        roll_quat = _rpy_to_quat_xyzw(0.0, 0.0, math.radians(roll_deg))
        quat_xyzw = _quat_multiply(quat_xyzw, roll_quat)
    
    return _normalize_quaternion(quat_xyzw)


# ---------------------------------------------------------------------------
# Main pose generation function
# ---------------------------------------------------------------------------

def generate_proto_poses(params: ProtoSimParameters) -> List[ProtoPose]:
    """
    Generate pose samples using spherical coordinate sampling.
    
    This mirrors the SpatialScout sampling pipeline:
    - For each (horiz, vert) combination, define a pivot point in the target frame
    - For each (pitch, yaw, distance, roll), compute camera position and orientation
    - Position is pivot + spherical_offset
    - Orientation points Z-axis at pivot with specified roll
    
    Args:
        params: Parameter ranges for sampling
        
    Returns:
        List of ProtoPose objects with positions and orientations in target object frame
    """
    poses: List[ProtoPose] = []
    
    for horiz in params.horiz:
        for vert in params.vert:
            # Pivot point in target frame (X=lateral, Y=forward, Z=vertical)
            pivot = (
                _mm_to_m(horiz),
                0.0,
                _mm_to_m(vert),
            )
            
            for pitch in params.pitch:
                for yaw in params.yaw:
                    for distance in params.distance:
                        for roll in params.roll:
                            # Compute position using spherical coordinates
                            offset = _spherical_to_cartesian(pitch, yaw, distance)
                            position = (
                                pivot[0] + offset[0],
                                pivot[1] + offset[1],
                                pivot[2] + offset[2],
                            )
                            
                            # Compute orientation (look at pivot with roll)
                            quat_xyzw = _look_at_quaternion(position, pivot, roll)
                            orientation_deg = _quat_to_rpy_deg(quat_xyzw)
                            
                            pose = ProtoPose(
                                parameters={
                                    "horiz": float(horiz),
                                    "vert": float(vert),
                                    "distance": float(distance),
                                    "roll": float(roll),
                                    "pitch": float(pitch),
                                    "yaw": float(yaw),
                                },
                                position_m=position,
                                orientation_deg=orientation_deg,
                                orientation_quat_xyzw=quat_xyzw,
                            )
                            poses.append(pose)
    
    return poses


def transform_pose_to_world(
    pose: ProtoPose,
    target_position: Tuple[float, float, float],
    target_orientation_quat_xyzw: Tuple[float, float, float, float],
) -> WorldPose:
    """
    Transform a pose from target object frame to the robot's base_link frame.
    
    The target_position and target_orientation should come from TF lookup
    of the target frame relative to base_link.
    
    Args:
        pose: Pose in target object's local frame
        target_position: Target object position relative to base_link (x, y, z) meters
        target_orientation_quat_xyzw: Target object orientation relative to base_link
        
    Returns:
        WorldPose with position and orientation in base_link frame
    """
    # Build rotation matrix from target orientation (xyzw quaternion)
    x, y, z, w = target_orientation_quat_xyzw
    
    # Quaternion to rotation matrix
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    rot = np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz), 2.0*(xz + wy)],
        [2.0*(xy + wz), 1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy), 2.0*(yz + wx), 1.0 - 2.0*(xx + yy)],
    ])
    
    # Transform position: p_ref = R_target * p_local + t_target
    p_local = np.array(pose.position_m)
    t_target = np.array(target_position)
    p_ref = rot @ p_local + t_target
    
    # Transform orientation: q_ref = q_target * q_local
    q_ref = _quat_multiply(target_orientation_quat_xyzw, pose.orientation_quat_xyzw)
    q_ref = _normalize_quaternion(q_ref)
    
    return WorldPose(
        name=pose.get_name(),
        position_m=tuple(p_ref),
        orientation_quat_xyzw=q_ref,
        parameters=pose.parameters,
    )


def generate_world_poses(
    params: ProtoSimParameters,
    target_position: Tuple[float, float, float],
    target_orientation_quat_xyzw: Tuple[float, float, float, float],
    randomize: bool = False,
) -> List[WorldPose]:
    """
    Generate poses directly in the robot's base_link frame.
    
    This is the main function for client-side pose generation:
    1. Generate poses in target object frame
    2. Transform all poses to base_link frame
    3. Optionally randomize order
    
    Args:
        params: Sampling parameters
        target_position: Target object position in base_link frame (from TF)
        target_orientation_quat_xyzw: Target object orientation in base_link frame
        randomize: Whether to randomize pose order
        
    Returns:
        List of WorldPose objects ready to send to server
    """
    # Generate poses in target frame
    local_poses = generate_proto_poses(params)
    
    # Transform to world (base_link) frame
    world_poses = [
        transform_pose_to_world(pose, target_position, target_orientation_quat_xyzw)
        for pose in local_poses
    ]
    
    # Optionally randomize
    if randomize:
        import random
        random.shuffle(world_poses)
    
    return world_poses


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def count_poses(params: ProtoSimParameters) -> int:
    """Count total number of poses that would be generated."""
    return (
        len(params.horiz) * 
        len(params.vert) * 
        len(params.distance) * 
        len(params.roll) * 
        len(params.pitch) * 
        len(params.yaw)
    )
