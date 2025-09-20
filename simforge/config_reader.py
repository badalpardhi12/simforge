"""Configuration models and YAML loader for Simforge.

Clean implementation without legacy baggage.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field


class SceneConfig(BaseModel):
    dt: float = 0.01
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    backend: str = "gpu"
    show_viewer: bool = True
    max_fps: int = 60


class ControlConfig(BaseModel):
    # execution
    joint_speed_limit: float = 1.0
    cartesian_speed_limit: float = 0.1
    cartesian_units: str = "m"  # "m" or "mm"
    # planning
    planner: str = "RRTConnect"
    planner_timeout: float = 3.0
    planner_resolution: float = 0.02
    planner_max_retry: int = 10
    cartesian_waypoints: int = 200
    strict_cartesian: bool = False
    # collision
    collision_check: bool = True
    self_collision_check: bool = True  # Allow disabling self-collision specifically
    min_clearance_m: float = 0.0
    collision_mesh_shrink: float = 1.0
    world_allowed_pairs: List[Tuple[str, str]] = Field(default_factory=list)
    ik_pos_tolerance_m: float = 1e-3  # 1 mm default
    ik_rot_tolerance_deg: float = 1.0  # 1 degree default
    ik_refine_pos_tolerance_m: float = 3e-4  # 0.3 mm corrective IK
    ik_refine_rot_tolerance_deg: float = 0.3  # 0.3° corrective IK
    ik_refine_max_attempts: int = 2
    # kinematics limits (fallbacks; prefer Pinocchio limits)
    max_joint_vel: float = 2.0
    max_joint_acc: float = 4.0
    # ground
    ground_plane_z: float = 0.0


class ToolConfig(BaseModel):
    urdf: str
    attach_link: Optional[str] = None  # Made optional - will use end_effector_link if not specified
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tcp_offset: Optional[Tuple[float, float, float, float, float, float, float]] = None

class RobotConfig(BaseModel):
    name: str
    urdf: str
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_base: bool = True
    initial_joint_positions: Optional[List[float]] = None
    end_effector_link: Optional[str] = None
    control: Optional[ControlConfig] = None
    parent: Optional[str] = None  # e.g., "obj:table1" or "robot:UR5e_1:wrist_3_link"
    tool: Optional[ToolConfig] = None


class ObjectConfig(BaseModel):
    type: str
    name: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Optional[Tuple[float, float, float]] = None
    collision_enabled: bool = True
    dynamic: bool = True


class ToolDefinition(BaseModel):
    """Global tool definition that can be referenced by robots."""
    name: str
    urdf: str
    tcp_offset: Optional[Tuple[float, float, float, float, float, float, float]] = None
    attach_offset: Optional[Tuple[float, float, float, float, float, float]] = None


class SimforgeConfig(BaseModel):
    scene: SceneConfig = Field(default_factory=SceneConfig)
    robots: List[RobotConfig] = Field(default_factory=list)
    objects: List[ObjectConfig] = Field(default_factory=list)
    control: ControlConfig = Field(default_factory=ControlConfig)
    tools: List[ToolDefinition] = Field(default_factory=list)

    def control_for(self, robot_name: str) -> ControlConfig:
        robot = next((r for r in self.robots if r.name == robot_name), None)
        if robot is None or robot.control is None:
            return self.control
        # Merge global with robot-specific
        base = self.control.model_dump()
        override = robot.control.model_dump()
        base.update(override)
        return ControlConfig(**base)

    @staticmethod
    def from_yaml(path: str | Path) -> "SimforgeConfig":
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}

        # Merge defaults.control → top-level control + each robot.control if not set
        defaults = data.get("defaults", {})
        default_ctrl = defaults.get("control", {}) if isinstance(defaults, dict) else {}
        if "control" in data:
            merged_top = {**default_ctrl, **(data["control"] or {})}
            data["control"] = merged_top
        elif default_ctrl:
            data["control"] = default_ctrl

        processed_robots = []
        for entry in data.get("robots", []):
            if isinstance(entry, str):
                inc_path = (path.parent / entry).resolve()
                inc_data = yaml.safe_load(inc_path.read_text()) or {}
                if "robots" in inc_data:
                    for r in inc_data["robots"]:
                        processed_robots.append(r)
                continue

            if isinstance(entry, dict) and "pose" in entry:
                pose = entry.pop("pose") or {}
                pos = pose.get("position")
                rpy = pose.get("rpy")
                if pos is not None:
                    entry["base_position"] = tuple(pos)
                if rpy is not None:
                    entry["base_orientation"] = tuple(rpy)
            # robot.control merge with defaults.control and global control
            global_ctrl = data.get("control", {})
            if default_ctrl or global_ctrl:
                rc = entry.get("control") or {}
                merged_ctrl = {**default_ctrl, **global_ctrl, **rc}
                entry["control"] = merged_ctrl

            processed_robots.append(entry)

        data["robots"] = processed_robots
        
        # Process tools if present
        tools = []
        for tool_data in data.get("tools", []):
            if isinstance(tool_data, dict):
                tools.append(tool_data)
        data["tools"] = tools
        
        return SimforgeConfig.model_validate(data)


__all__ = ["SimforgeConfig", "RobotConfig", "ControlConfig", "SceneConfig", "ObjectConfig", "ToolConfig", "ToolDefinition"]
