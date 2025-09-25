"""Genesis joint I/O helpers."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

import logging


def _ensure_length(values: Iterable[float], size: int, pad: float = 0.0) -> list[float]:
    data = list(values)
    if len(data) > size:
        return data[:size]
    if len(data) < size:
        data.extend([pad] * (size - len(data)))
    return data


def get_joint_positions(entity, logger: Optional[logging.Logger] = None, prefer_struct: bool = False) -> np.ndarray:
    logger = logger or logging.getLogger("simforge.genesis")
    try:
        if prefer_struct and hasattr(entity, "get_dofs"):
            dofs = entity.get_dofs()
            pos = getattr(dofs, "position", None)
            if pos is not None:
                if hasattr(pos, "detach"):
                    pos = pos.detach().cpu().numpy()
                elif hasattr(pos, "cpu"):
                    pos = pos.cpu().numpy()
                return np.asarray(pos, dtype=np.float64)
        if hasattr(entity, "get_dofs_position"):
            values = entity.get_dofs_position()
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            elif hasattr(values, "cpu"):
                values = values.cpu().numpy()
            return np.asarray(values, dtype=np.float64)
        logger.error("Genesis entity exposes no readable DOF interface; returning zeros.")
        return np.zeros(0, dtype=np.float64)
    except Exception as exc:
        logger.error("Failed to read Genesis DOFs: %s", exc)
        return np.zeros(0, dtype=np.float64)


def set_joint_positions(entity, joints: Iterable[float], expected_dofs: int, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or logging.getLogger("simforge.genesis")
    command = _ensure_length(joints, expected_dofs)

    def _call(method: str) -> bool:
        if not hasattr(entity, method):
            return False
        fn = getattr(entity, method)
        try:
            fn(command, list(range(len(command))))
            return True
        except TypeError:
            try:
                fn(command)
                return True
            except Exception as exc:  # pragma: no cover - debug path
                logger.debug("%s invocation failed: %s", method, exc)
                return False
        except Exception as exc:
            logger.debug("%s invocation failed: %s", method, exc)
            return False

    applied = False
    for name in ("set_dofs_position", "set_qpos", "set_q"):
        if _call(name):
            applied = True
            break

    for name in ("control_dofs_position", "set_dofs_position_target", "set_dofs_target"):
        if _call(name):
            applied = True

    if not applied:
        logger.error("No applicable Genesis setter found; joints not applied.")

    current = get_joint_positions(entity, logger, prefer_struct=True)
    if current.size:
        logger.debug("set -> %s; read -> %s", np.round(command, 5), np.round(current, 5))


__all__ = ["get_joint_positions", "set_joint_positions"]
