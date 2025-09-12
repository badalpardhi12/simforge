"""Configuration models and YAML loader for Simforge.

Clean implementation without legacy baggage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field


class SceneConfig(BaseModel):
    dt: float = 0.01
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    backend: str = "gpu"
    show_viewer: bool = True
    max_fps: int = 60


class ControlConfig(BaseModel):
    joint_speed_limit: float = 1.0
    cartesian_speed_limit: float = 0.1
    collision_check: bool = True
    planner_timeout: float = 1.0
    max_joint_vel: float = 2.0
    max_joint_acc: float = 4.0


class RobotConfig(BaseModel):
    name: str
    urdf: str
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_base: bool = True
    initial_joint_positions: Optional[List[float]] = None
    end_effector_link: Optional[str] = None
    control: Optional[ControlConfig] = None


class ObjectConfig(BaseModel):
    type: str
    name: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Optional[Tuple[float, float, float]] = None
    collision_enabled: bool = True


class SimforgeConfig(BaseModel):
    scene: SceneConfig = Field(default_factory=SceneConfig)
    robots: List[RobotConfig] = Field(default_factory=list)
    objects: List[ObjectConfig] = Field(default_factory=list)
    control: ControlConfig = Field(default_factory=ControlConfig)

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
        
        # Process robot includes
        processed_robots = []
        for entry in data.get("robots", []):
            if isinstance(entry, str):
                # Include from file
                inc_path = (path.parent / entry).resolve()
                inc_data = yaml.safe_load(inc_path.read_text()) or {}
                if "robots" in inc_data:
                    processed_robots.extend(inc_data["robots"])
            else:
                # Support legacy 'pose' key mapping to base_position/base_orientation (rpy)
                if isinstance(entry, dict) and "pose" in entry:
                    pose = entry.pop("pose") or {}
                    pos = pose.get("position")
                    rpy = pose.get("rpy")
                    if pos is not None:
                        entry["base_position"] = tuple(pos)
                    if rpy is not None:
                        entry["base_orientation"] = tuple(rpy)
                processed_robots.append(entry)
        
        data["robots"] = processed_robots
        return SimforgeConfig.model_validate(data)


__all__ = ["SimforgeConfig", "RobotConfig", "ControlConfig", "SceneConfig", "ObjectConfig"]