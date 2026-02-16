"""Environment configuration loader for the Simforge schema."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional
import xml.etree.ElementTree as ET

import yaml

try:  # pragma: no cover - optional dependency path
    from simforge.tooling import ToolDefinition, ToolManager
except ModuleNotFoundError:  # pragma: no cover - fallback when legacy tooling absent
    from dataclasses import dataclass

    @dataclass
    class ToolDefinition:  # type: ignore[misc]
        name: str
        urdf_path: str
        tcp_offset: tuple[float, ...] | None = None
        attach_offset: tuple[float, ...] | None = None

    class ToolManager:  # type: ignore[misc]
        def __init__(self, logger_obj: logging.Logger) -> None:
            self._logger = logger_obj
            self._tools: Dict[str, ToolDefinition] = {}

        def register_tool(self, tool: ToolDefinition) -> None:
            self._tools[tool.name] = tool
            if not Path(tool.urdf_path).exists():
                self._logger.warning(
                    "Tool URDF '%s' not found; attachment may fall back to base robot.",
                    tool.urdf_path,
                )

        def attach_tool_to_robot_urdf(
            self,
            robot_urdf_path: str,
            tool_name: str,
            end_effector_link: str,
            output_path: str,
        ) -> str:
            tool = self._tools.get(tool_name)
            base_path = Path(robot_urdf_path).resolve()
            out_path = Path(output_path).resolve()

            if tool is None:
                self._logger.warning(
                    "Tool '%s' was not registered; skipping attachment for '%s'.",
                    tool_name,
                    base_path,
                )
                return robot_urdf_path

            if _combine_urdf_with_tool(
                base_path,
                Path(tool.urdf_path).resolve(),
                out_path,
                end_effector_link,
                tool.attach_offset,
                self._logger,
            ):
                self._logger.info(
                    "Attached tool '%s' to URDF '%s' -> '%s'",
                    tool_name,
                    base_path,
                    out_path,
                )
                return str(out_path)

            self._logger.warning(
                "Could not attach tool '%s'; using base URDF '%s'.",
                tool_name,
                base_path,
            )
            return robot_urdf_path

from ..core.config_schema import EnvironmentConfig as EnvironmentDocument, WorldObjectType
from ..core.models import EnvironmentSpec, build_robot_profile

logger = logging.getLogger(__name__)


def _combine_urdf_with_tool(
    base_urdf: Path,
    tool_urdf: Path,
    output_path: Path,
    parent_link: str,
    attach_offset: tuple[float, ...] | None,
    logger_obj: logging.Logger,
) -> bool:
    """Combine robot and tool URDFs into a single file."""
    try:
        base_tree = ET.parse(base_urdf)
        base_root = base_tree.getroot()
        tool_tree = ET.parse(tool_urdf)
        tool_root = tool_tree.getroot()
    except Exception as exc:  # pragma: no cover - defensive
        logger_obj.warning("Failed to parse URDFs for attachment: %s", exc)
        return False

    prefix = tool_urdf.stem

    def _prefixed(name: str) -> str:
        return f"{prefix}::{name}"

    # Collect existing names to avoid clashes
    existing_links = {elem.get("name") for elem in base_root.findall("link")}
    existing_joints = {elem.get("name") for elem in base_root.findall("joint")}

    # Determine tool base link
    tool_links = [elem.get("name") for elem in tool_root.findall("link")]
    child_links = {child.get("link") for child in tool_root.findall("joint/child")}
    if not tool_links:
        logger_obj.warning("Tool URDF '%s' has no links", tool_urdf)
        return False
    root_link = next((name for name in tool_links if name not in child_links), tool_links[0])

    # Copy materials (if any)
    for material in tool_root.findall("material"):
        base_root.append(copy.deepcopy(material))

    # Copy links with prefixes
    for link_elem in tool_root.findall("link"):
        link_copy = copy.deepcopy(link_elem)
        name = link_copy.get("name")
        if not name:
            continue
        new_name = _prefixed(name)
        if new_name in existing_links:
            logger_obj.warning("Link name collision '%s'; skipping", new_name)
            continue
        link_copy.set("name", new_name)
        base_root.append(link_copy)
        existing_links.add(new_name)

    # Copy joints with prefixes
    for joint_elem in tool_root.findall("joint"):
        joint_copy = copy.deepcopy(joint_elem)
        name = joint_copy.get("name")
        if name is None:
            name = f"{prefix}_joint_{len(existing_joints)}"
        new_name = _prefixed(name)
        if new_name in existing_joints:
            logger_obj.warning("Joint name collision '%s'; skipping", new_name)
            continue
        joint_copy.set("name", new_name)
        parent_elem = joint_copy.find("parent")
        if parent_elem is not None and parent_elem.get("link"):
            parent_elem.set("link", _prefixed(parent_elem.get("link")))
        child_elem = joint_copy.find("child")
        if child_elem is not None and child_elem.get("link"):
            child_elem.set("link", _prefixed(child_elem.get("link")))
        base_root.append(joint_copy)
        existing_joints.add(new_name)

    # Create attachment joint
    root_link_prefixed = _prefixed(root_link)
    if root_link_prefixed not in existing_links:
        logger_obj.warning("Tool root link '%s' missing after copy", root_link_prefixed)
        return False

    attachment = ET.Element("joint", name=f"{prefix}_attachment", type="fixed")
    origin = ET.SubElement(attachment, "origin")
    if attach_offset is not None:
        px, py, pz, roll, pitch, yaw = attach_offset
    else:
        px = py = pz = roll = pitch = yaw = 0.0
    origin.set("xyz", f"{px} {py} {pz}")
    origin.set("rpy", f"{roll} {pitch} {yaw}")

    parent_elem = ET.SubElement(attachment, "parent")
    parent_elem.set("link", parent_link)

    child_elem = ET.SubElement(attachment, "child")
    child_elem.set("link", root_link_prefixed)

    base_root.append(attachment)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return True
    except Exception as exc:  # pragma: no cover - filesystem issues
        logger_obj.warning("Failed to write combined URDF '%s': %s", output_path, exc)
        return False


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
    
    # First, try direct resolution from base_dir (handles ../../../ style paths)
    direct_resolved = (base_dir / candidate).resolve()
    if direct_resolved.exists():
        return str(direct_resolved)
    
    # Extract the meaningful part of the path (e.g., "assets/ur5e/ur5e.urdf" from "../../../assets/...")
    path_parts = Path(path).parts
    # Find 'assets' in path and use from there
    meaningful_path = None
    for i, part in enumerate(path_parts):
        if part == "assets":
            meaningful_path = Path(*path_parts[i:])
            break
    
    # Search roots: base_dir parents, package root, cwd
    search_roots = [base_dir, *base_dir.parents]
    
    # Also try from the simforge_new package root
    package_root = Path(__file__).resolve().parents[2]  # simforge_new → repo root
    simforge_new_root = Path(__file__).resolve().parents[1]  # simforge_new/
    if simforge_new_root not in search_roots:
        search_roots.append(simforge_new_root)
    if package_root not in search_roots:
        search_roots.append(package_root)
    
    # Add current working directory
    cwd = Path.cwd()
    if cwd not in search_roots:
        search_roots.append(cwd)
    
    for root in search_roots:
        # Try the meaningful path (assets/...) directly under root
        if meaningful_path:
            resolved = (root / meaningful_path).resolve()
            if resolved.exists():
                return str(resolved)
        
        # Try the original candidate
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    
    # If nothing found, return the meaningful path resolved from package root as best guess
    if meaningful_path:
        return str((package_root / meaningful_path).resolve())
    
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
