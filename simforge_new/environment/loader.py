"""Environment configuration loader for the Simforge schema."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import yaml

from simforge.tooling import ToolDefinition, ToolManager

from ..core.config_schema import EnvironmentConfig as EnvironmentDocument, WorldObjectType
from ..core.models import EnvironmentSpec, build_robot_profile

logger = logging.getLogger(__name__)


class EnvironmentLoadError(RuntimeError):
    """Raised when an environment preset cannot be loaded."""


def _merge_dicts(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            _merge_dicts(base[key], value)
        elif key in base and isinstance(base[key], list) and isinstance(value, (list, tuple)):
            base[key] = list(base[key]) + list(value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:  # pragma: no cover - propagate IO/parse errors directly
        raise EnvironmentLoadError(f"Failed to read {path}: {exc}") from exc
    return data or {}


def _resolve_path(path: Optional[str], base_dir: Path) -> Optional[str]:
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    search_roots = [base_dir, *base_dir.parents]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str((base_dir / candidate).resolve())


def _resolve_includes(path: Path, visited: set[Path] | None = None) -> Dict[str, Any]:
    visited = visited or set()
    if path in visited:
        raise EnvironmentLoadError(f"Cyclic include detected at {path}")
    visited.add(path)

    data = _load_yaml(path)
    includes = list(data.get("includes", []) or [])
    merged: MutableMapping[str, Any] = {}
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        inc_data = _resolve_includes(inc_path, visited)
        merged = _merge_dicts(merged, inc_data)
    merged = _merge_dicts(merged, data)
    return dict(merged)


def load_environment(path: str | Path) -> EnvironmentSpec:
    path = Path(path).resolve()
    raw = _resolve_includes(path)
    try:
        config = EnvironmentDocument.model_validate(raw)
    except Exception as exc:
        raise EnvironmentLoadError(f"Invalid configuration at {path}: {exc}") from exc

    robots = []
    tool_manager: ToolManager | None = None

    for spec in config.robots:
        if not spec.urdf:
            raise EnvironmentLoadError(f"Robot '{spec.name}' missing URDF path")

        resolved_urdf = _resolve_path(spec.urdf, path.parent)
        tool = spec.tool
        if tool and tool.urdf_override:
            resolved_tool = tool.model_copy(update={"urdf_override": _resolve_path(tool.urdf_override, path.parent)})
            spec = spec.model_copy(update={"tool": resolved_tool})

        metadata = dict(spec.metadata or {})

        if spec.tool and spec.tool.urdf_override:
            if tool_manager is None:
                tool_manager = ToolManager(logger)
            tool_name = spec.tool.name or f"{spec.name}_tool"
            attach_offset = None
            if spec.tool.attach_pose:
                px, py, pz, roll, pitch, yaw = spec.tool.attach_pose
                attach_offset = (
                    float(px),
                    float(py),
                    float(pz),
                    math.radians(float(roll)),
                    math.radians(float(pitch)),
                    math.radians(float(yaw)),
                )
            tcp_offset = tuple(float(v) for v in spec.tool.tcp_offset) if spec.tool.tcp_offset else None
            tool_def = ToolDefinition(
                name=tool_name,
                urdf_path=str(spec.tool.urdf_override),
                tcp_offset=tcp_offset,
                attach_offset=attach_offset,
            )
            tool_manager.register_tool(tool_def)
            base_urdf_path = Path(resolved_urdf)
            combined_path = base_urdf_path.parent / f"{base_urdf_path.stem}_with_{tool_def.name}.urdf"
            try:
                resolved_urdf = tool_manager.attach_tool_to_robot_urdf(
                    resolved_urdf,
                    tool_def.name,
                    spec.end_effector_link or "",
                    str(combined_path),
                )
                metadata.setdefault("tool_name", tool_name)
                metadata.setdefault("tool_urdf", resolved_urdf)
            except Exception as exc:  # pragma: no cover - relies on external tooling
                logger.warning("Failed to attach tool '%s' to %s: %s", tool_name, spec.name, exc)

        spec_with_metadata = spec.model_copy(update={"metadata": {k: str(v) for k, v in metadata.items()}})
        profile = build_robot_profile(spec_with_metadata, resolved_urdf or "", spec.end_effector_link or "")
        robots.append(profile)

    resolved_world_objects = []
    for obj in config.world.objects:
        if obj.type == WorldObjectType.URDF and obj.urdf:
            resolved_obj = obj.model_copy(update={"urdf": _resolve_path(obj.urdf, path.parent)})
            resolved_world_objects.append(resolved_obj)
        else:
            resolved_world_objects.append(obj)

    world_config = config.world.model_copy(update={"objects": resolved_world_objects})

    return EnvironmentSpec(
        config=config,
        scene=config.scene,
        world=world_config,
        robots=tuple(robots),
    )


__all__ = ["EnvironmentLoadError", "load_environment"]
