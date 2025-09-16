"""Genesis robot I/O helpers."""
from __future__ import annotations

from typing import List, Optional
import numpy as np

from .utils import ensure_length


def apply_joint_positions(entity, q_rad: List[float], logger) -> bool:
    """Best-effort joint application across different Genesis builds."""
    applied = False
    dofs_idx = list(range(len(q_rad)))

    def _call(method: str) -> bool:
        if not hasattr(entity, method):
            return False
        fn = getattr(entity, method)
        try:
            fn(q_rad, dofs_idx)
            return True
        except TypeError:
            try:
                fn(q_rad)
                return True
            except Exception as exc2:
                logger.debug(f"{method}(values) failed: {exc2}")
                return False
        except Exception as exc:
            logger.debug(f"{method}(values, idx) failed: {exc}")
            return False

    for name in ("set_dofs_position", "set_qpos", "set_q"):
        if _call(name):
            applied = True
            break

    for name in ("control_dofs_position", "set_dofs_position_target", "set_dofs_target"):
        if _call(name):
            applied = True

    return applied


def set_robot_joints(entity, q_rad: List[float], expected_dofs: int, logger) -> None:
    """Normalize and apply the joint vector to the Genesis entity."""
    if not q_rad or any(not np.isfinite(q) for q in q_rad):
        logger.warning(f"Invalid joint vector {q_rad}")
        return

    command = ensure_length(q_rad, expected_dofs)
    if not apply_joint_positions(entity, command, logger):
        logger.error("No applicable Genesis setter found; joints not applied.")
        return

    verification = get_robot_joints(entity, logger, prefer_struct=True)
    logger.debug(f"set -> {np.round(command, 5)} ; read -> {np.round(verification, 5)}")


def get_robot_joints(entity, logger, prefer_struct: bool = False) -> np.ndarray:
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

        if hasattr(entity, "get_dofs_position"):
            jp = entity.get_dofs_position()
            if hasattr(jp, "cpu"):
                jp = jp.cpu().numpy()
            elif hasattr(jp, "detach"):
                jp = jp.detach().cpu().numpy()
            return np.array(jp, dtype=np.float32)

        logger.error("Genesis entity exposes no readable DOF interface; returning zeros.")
        return np.zeros(6, dtype=np.float32)
    except Exception as exc:
        logger.error(f"Failed to read Genesis DOFs: {exc}")
        return np.zeros(6, dtype=np.float32)


__all__ = ["set_robot_joints", "get_robot_joints", "apply_joint_positions"]
