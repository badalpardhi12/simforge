"""Quaternion and rotation utilities used across the control stack."""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll/pitch/yaw (radians) to quaternion (w, x, y, z)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = np.array([w, x, y, z], dtype=np.float64)
    return _normalize(quat)


def quaternion_multiply(q1: Iterable[float], q2: Iterable[float]) -> np.ndarray:
    """Hamilton product of quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = (float(v) for v in q1)
    w2, x2, y2, z2 = (float(v) for v in q2)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return _normalize(np.array([w, x, y, z], dtype=np.float64))


def quaternion_to_rotation_matrix(quat: Iterable[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix for quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in quat)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot = np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )
    return rot


def _normalize(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


__all__ = [
    "rpy_to_quaternion",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
]
