from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field


# Core scene/viewer options
class SceneConfig(BaseModel):
    dt: float = 0.01
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    backend: str = "gpu"  # "gpu" (auto), "cpu", "cuda"
    show_viewer: bool = True
    max_fps: int = 60


# Robot descriptor (URDF + placement + optional EE and per-robot overrides)
class RobotConfig(BaseModel):
    name: str
    urdf: str
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fixed_base: bool = True
    initial_joint_positions: Optional[List[float]] = None  # degrees
    kp: Optional[List[float]] = None
    kv: Optional[List[float]] = None
    force_limits: Optional[Tuple[List[float], List[float]]] = None
    end_effector_link: Optional[str] = None
    # Optional: per-robot control overrides (merged over globals by control_for)
    control: Optional["ControlConfig"] = None


# Simple environment object
class ObjectConfig(BaseModel):
    type: str  # "plane", "box" (future: "mesh")
    name: str = ""  # optional unique name for ACM / debugging
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees
    size: Optional[Tuple[float, float, float]] = None  # for box
    collision_enabled: bool = True


# Planning + collision + execution parameters
class ControlConfig(BaseModel):
    joint_speed_limit: float = 1.0
    cartesian_speed_limit: float = 0.1
    default_tcp: Tuple[float, float, float, float, float, float, float] = (
        0,
        0,
        0,
        1,
        0,
        0,
        0,
    )
    strict_cartesian: bool = True

    # Collision control
    collision_check: bool = True
    ccd_substeps: int = 5
    allowed_collision_links: List[Tuple[str, str]] = Field(default_factory=list)
    ground_plane_z: float = 0.0
    collision_mesh_shrink: float = 1.0
    auto_allow_home_collisions: bool = True
    visualize_collision_meshes: bool = False

    # Planning control
    planner: str = "RRTConnect"
    planner_timeout: float = 1.0
    planner_resolution: float = 0.02
    planner_max_retry: int = 50
    cartesian_waypoints: int = 100
    postcheck_time_s: float = 0.2

    # World-collision options (for inter-robot and static objects)
    min_clearance_m: float = 0.005
    world_allowed_pairs: List[Tuple[str, str]] = Field(
        default_factory=list
    )  # e.g., ["ur5e_1/wrist_3_link", "obj:table1"]


class SimforgeConfig(BaseModel):
    """
    Top-level configuration container for a Simforge simulation.
    Combines scene settings, robot definitions, environment objects, and control parameters.

    Attributes:
        scene: Simulation scene parameters (backend, FPS, gravity)
        robots: List of robots to load with their URDFs and configurations
        objects: List of static objects (planes, boxes) in the scene
        control: Global control settings; can be overridden per robot
    """

    scene: SceneConfig = Field(default_factory=SceneConfig)
    robots: List[RobotConfig] = Field(default_factory=list)
    objects: List[ObjectConfig] = Field(default_factory=list)
    control: ControlConfig = Field(default_factory=ControlConfig)

    # Return effective control for a robot: per-robot override merged over globals.
    def control_for(self, robot_name: str) -> ControlConfig:
        # Find the robot entry
        rob = next((r for r in self.robots if r.name == robot_name), None)
        if rob is None or rob.control is None:
            return self.control
        # Merge global (base) with per-robot override
        base = self.control.model_dump()
        override = rob.control.model_dump()
        base.update(override)
        return ControlConfig(**base)

    @staticmethod
    def from_yaml(path: str | Path) -> "SimforgeConfig":
        """
        Flexible loader:
        - Supports top-level 'defaults.control' merged with top-level 'control' for global defaults
        - Supports robots entries as:
          * inline mapping
          * {'include': 'file.yaml', ...overrides...} (extracts the first robot from included file)
          * string path to a yaml (equivalent to {'include': path})
        - Supports 'pose': {position:[x,y,z], rpy:[r,p,y]} override which maps to base_position/base_orientation
        - Per-robot 'control' overrides are stored and merged at access time by control_for()
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}

        # Global control = defaults.control overlaid by top-level control
        defaults_map = data.get("defaults", {})
        defaults_ctrl = defaults_map.get("control", {})
        top_ctrl = data.get("control", {})
        gctrl = {**defaults_ctrl, **top_ctrl}
        data["control"] = gctrl

        # Process robot includes
        processed_robots = []
        for entry in data.get("robots", []):
            if isinstance(entry, str):
                inc_path = (path.parent / entry).resolve()
                inc_data = yaml.safe_load(inc_path.read_text()) or {}
                if not inc_data.get("robots"):
                    raise ValueError(
                        f"Included file {inc_path} has no 'robots' list"
                    )
                processed_robots.append(inc_data["robots"][0])
            elif isinstance(entry, dict) and "include" in entry:
                inc_path = (path.parent / str(entry["include"])).resolve()
                inc_data = yaml.safe_load(inc_path.read_text()) or {}
                if not inc_data.get("robots"):
                    raise ValueError(
                        f"Included file {inc_path} has no 'robots' list"
                    )
                base_map = inc_data["robots"][0]
                overlay = {k: v for k, v in entry.items() if k != "include"}
                base_map.update(overlay)
                processed_robots.append(base_map)
            else:
                # Handle 'pose' key mapping to base_position/orientation
                if 'pose' in entry and isinstance(entry['pose'], dict):
                    pose_data = entry.pop('pose')
                    if 'position' in pose_data:
                        entry['base_position'] = pose_data['position']
                    if 'rpy' in pose_data:
                        entry['base_orientation'] = pose_data['rpy']
                processed_robots.append(entry)
        data["robots"] = processed_robots

        return SimforgeConfig.model_validate(data)
