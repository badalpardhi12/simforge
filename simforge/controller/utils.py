"""Utility helpers shared across movement controller modules."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import numpy as np


def deg_to_rad_list(values: Sequence[float]) -> List[float]:
    """Convert a sequence of degrees to radians."""
    return [float(np.deg2rad(v)) for v in values]


def rad_to_deg_list(values: Sequence[float]) -> List[float]:
    """Convert a sequence of radians to degrees."""
    return [float(np.rad2deg(v)) for v in values]


def clamp_vector(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clamp ``values`` to lie within ``[lower, upper]``."""
    return np.clip(values, lower, upper)


def ensure_length(values: Sequence[float], size: int, pad_value: float = 0.0) -> List[float]:
    """Trim or pad the list to ``size`` elements."""
    data = list(values)
    if len(data) > size:
        return data[:size]
    if len(data) < size:
        return data + [pad_value] * (size - len(data))
    return data


def as_float_array(values: Iterable[float], size: int | None = None, dtype=np.float64) -> np.ndarray:
    """Convert ``values`` to a NumPy array with optional size adjustment."""
    arr = np.array(list(values), dtype=dtype)
    if size is not None and arr.size != size:
        tmp = np.zeros(size, dtype=dtype)
        n = min(size, arr.size)
        if n > 0:
            tmp[:n] = arr[:n]
        arr = tmp
    return arr


def to_meters(ctrl_config, pos_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert GUI units to meters depending on control config."""
    units = (getattr(ctrl_config, "cartesian_units", "m") or "m").lower()
    if units == "mm":
        return tuple(float(v) / 1000.0 for v in pos_xyz)
    return tuple(float(v) for v in pos_xyz)


__all__ = [
    "deg_to_rad_list",
    "rad_to_deg_list",
    "clamp_vector",
    "ensure_length",
    "as_float_array",
    "to_meters",
]
