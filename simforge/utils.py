"""Utility functions for Simforge to reduce code duplication."""

from __future__ import annotations

import numpy as np
from typing import Any, Optional


def to_numpy(tensor_or_array: Any) -> np.ndarray:
    """
    Convert tensor or array-like to numpy array, handling PyTorch tensors and similar.
    
    Args:
        tensor_or_array: Input tensor/array to convert
        
    Returns:
        np.ndarray: Converted array
    """
    if hasattr(tensor_or_array, 'detach'):
        # PyTorch tensor
        return tensor_or_array.detach().cpu().numpy()
    elif hasattr(tensor_or_array, 'numpy'):
        # Other tensor types with .numpy() method
        return tensor_or_array.numpy()
    else:
        # Already numpy or convertible
        return np.array(tensor_or_array)


def safe_set_dofs_position(entity, q_rad: np.ndarray, dofs_idx: Optional[list[int]] = None) -> None:
    """
    Safely set degrees of freedom positions with fallback for different Genesis API versions.
    
    Args:
        entity: Genesis entity
        q_rad: Joint positions in radians
        dofs_idx: Degrees of freedom indices (optional)
    """
    try:
        if dofs_idx is None:
            entity.set_dofs_position(q_rad)
        else:
            entity.set_dofs_position(q_rad, dofs_idx_local=dofs_idx)
    except TypeError:
        # Fallback for older API
        entity.set_dofs_position(q_rad, dofs_idx=dofs_idx)


def rpy_to_quat_wxyz(r: float, p: float, y: float) -> np.ndarray:
    """
    Convert roll, pitch, yaw (radians) to quaternion in WXYZ format.
    
    Args:
        r: roll (radians)
        p: pitch (radians)  
        y: yaw (radians)
        
    Returns:
        np.ndarray: Quaternion [qw, qx, qy, qz]
    """
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    return np.array([
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        sy*cp*sr + cy*sp*cr,
        sy*cp*cr - cy*sp*sr,
    ], dtype=np.float32)


def quat_wxyz_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions in WXYZ format.
    
    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]
        
    Returns:
        np.ndarray: Result quaternion [qw, qx, qy, qz]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], dtype=np.float32)


def quat_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [qw, qx, qy, qz]
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def quat_wxyz_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector by quaternion.
    
    Args:
        q: Quaternion [qw, qx, qy, qz]
        v: Vector [vx, vy, vz]
        
    Returns:
        np.ndarray: Rotated vector [vx', vy', vz']
    """
    return quat_wxyz_to_rotation_matrix(q) @ v


def quat_wxyz_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion in WXYZ format."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_wxyz_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion."""
    n = np.linalg.norm(q)
    return q / (n + 1e-9)