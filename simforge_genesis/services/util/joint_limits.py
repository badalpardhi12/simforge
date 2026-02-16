"""Helpers for resolving and enforcing joint limits."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from ...core.models import RobotProfile


def resolve_joint_limits(
    profile: RobotProfile,
    entity: Optional[object] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return joint lower/upper limits as numpy arrays (radians).

    Order matches the profile joint indexing. Returns ``(None, None)`` when
    limits cannot be determined.
    """

    metadata = profile.metadata or {}
    limits = metadata.get("joint_limits_rad")
    if limits:
        lower, upper = _normalize_metadata_limits(limits)
        if lower is not None and upper is not None:
            return lower, upper

    lower = _array_from_entity(entity, "qpos_lower")
    upper = _array_from_entity(entity, "qpos_upper")
    if lower is not None and upper is not None and lower.size == upper.size:
        return lower, upper

    lower = _array_from_entity(entity, "joint_lower_limit")
    upper = _array_from_entity(entity, "joint_upper_limit")
    if lower is not None and upper is not None and lower.size == upper.size:
        return lower, upper

    urdf_limits = _parse_urdf_limits(profile)
    if urdf_limits is not None:
        return urdf_limits

    return None, None


def wrap_vector_to_limits(
    values: np.ndarray,
    reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Wrap each joint angle inside limits while staying close to reference."""

    if lower is None or upper is None:
        return values
    wrapped = values.copy()
    for idx in range(min(len(wrapped), len(lower))):
        wrapped[idx] = wrap_angle_to_limits(
            wrapped[idx],
            reference[idx] if reference is not None and reference.size > idx else wrapped[idx],
            lower[idx],
            upper[idx],
        )
    return wrapped


def wrap_angle_to_limits(value: float, reference: float, lower: float, upper: float) -> float:
    """Wrap a revolute joint angle into [lower, upper] near ``reference``."""

    if not (math.isfinite(lower) and math.isfinite(upper)) or upper <= lower:
        return value

    period = 2.0 * math.pi
    span = upper - lower
    candidates = []
    if span >= period - 1e-9:
        k_min = math.floor((lower - value) / period) - 1
        k_max = math.ceil((upper - value) / period) + 1
    else:
        k_min = math.floor((lower - value) / period)
        k_max = math.ceil((upper - value) / period)

    for k in range(k_min, k_max + 1):
        candidate = value + k * period
        if lower - 1e-9 <= candidate <= upper + 1e-9:
            candidates.append(candidate)

    if not candidates:
        return min(max(value, lower), upper)

    if reference is None or not math.isfinite(reference):
        reference = value

    best = min(candidates, key=lambda c: abs(c - reference))
    return float(best)


def _normalize_metadata_limits(
    limits: Sequence[Sequence[float]] | Sequence[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        arr = np.asarray(limits, dtype=np.float64)
    except Exception:
        return None, None
    if arr.ndim == 2 and arr.shape[1] >= 2:
        lower = arr[:, 0]
        upper = arr[:, 1]
        return lower, upper
    return None, None


def _array_from_entity(entity: Optional[object], attr: str) -> Optional[np.ndarray]:
    if entity is None or not hasattr(entity, attr):
        return None
    value = getattr(entity, attr)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ModuleNotFoundError:
        pass
    if hasattr(value, "__array__"):
        return np.asarray(value, dtype=np.float64)
    return None


def _parse_urdf_limits(profile: RobotProfile) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        urdf_path = Path(profile.urdf)
    except Exception:
        return None
    candidates = [urdf_path]
    if not urdf_path.is_absolute():
        candidates.append(Path.cwd() / urdf_path)
        candidates.append(Path(__file__).resolve().parents[3] / urdf_path)

    target_path = next((p for p in candidates if p.exists()), None)
    if target_path is None:
        return None

    try:
        root = ET.parse(target_path).getroot()
    except Exception:
        return None

    lower: list[float] = []
    upper: list[float] = []
    for joint in root.findall("joint"):
        joint_type = joint.get("type", "revolute").lower()
        if joint_type not in ("revolute", "continuous"):
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        try:
            lo = float(limit.get("lower", "-inf"))
            hi = float(limit.get("upper", "inf"))
        except (TypeError, ValueError):
            continue
        lower.append(lo)
        upper.append(hi)
        if len(lower) >= profile.joint_count:
            break

    if len(lower) < profile.joint_count:
        return None

    return np.asarray(lower[: profile.joint_count], dtype=np.float64), np.asarray(upper[: profile.joint_count], dtype=np.float64)


__all__ = [
    "resolve_joint_limits",
    "wrap_angle_to_limits",
    "wrap_vector_to_limits",
]

