"""Inverse kinematics solver using Pinocchio.

Clean IK implementation without the complexities of the legacy simulator.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False


def solve_ik(
    model: "pin.Model",
    data: "pin.Data", 
    frame_name: str,
    q_init: np.ndarray,
    target_pos: tuple,
    target_quat_wxyz: tuple,
    max_iters: int = 100,
    pos_tolerance: float = 1e-3,
    rot_tolerance_deg: float = 1.0,
    damping: float = 1e-3
) -> Optional[np.ndarray]:
    """Solve inverse kinematics using damped least squares.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        frame_name: Target frame name
        q_init: Initial joint configuration
        target_pos: Target position (x, y, z)
        target_quat_wxyz: Target quaternion (w, x, y, z)
        max_iters: Maximum iterations
        pos_tolerance: Position tolerance in meters
        rot_tolerance_deg: Rotation tolerance in degrees
        damping: Damping factor for DLS
        
    Returns:
        Joint configuration or None if failed
    """
    if not HAS_PINOCCHIO:
        return None
        
    try:
        frame_id = model.getFrameId(frame_name)
    except:
        return None

    # Setup target pose
    target_translation = np.array(target_pos, dtype=np.float64)
    qw, qx, qy, qz = target_quat_wxyz
    target_quat = pin.Quaternion(qw, qx, qy, qz)
    target_quat.normalize()
    target_pose = pin.SE3(target_quat.toRotationMatrix(), target_translation)

    q = q_init.astype(np.float64).copy()
    rot_tol = np.deg2rad(rot_tolerance_deg)

    for _ in range(max_iters):
        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_pose = data.oMf[frame_id]

        # Compute error
        pose_error = pin.log6(current_pose.inverse() * target_pose)
        pos_error = pose_error.linear
        rot_error = pose_error.angular

        # Check convergence
        if np.linalg.norm(pos_error) < pos_tolerance and np.linalg.norm(rot_error) < rot_tol:
            return q.astype(np.float32)

        # Compute Jacobian
        pin.computeJointJacobians(model, data, q)
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # Damped least squares
        error_vec = np.concatenate([rot_error, pos_error])
        JJT = J @ J.T + damping * np.eye(6)
        try:
            dnu = np.linalg.solve(JJT, error_vec)
            dq = J.T @ dnu
        except np.linalg.LinAlgError:
            return None

        # Update configuration with step limiting
        step_size = min(1.0, 0.1 / max(np.abs(dq).max(), 1e-6))
        q += step_size * dq

    return None


__all__ = ["solve_ik", "HAS_PINOCCHIO"]