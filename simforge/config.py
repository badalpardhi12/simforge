from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class SceneConfig:
    dt: float = 0.01
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    backend: str = "gpu"  # "gpu" (auto), "cpu", "cuda"
    show_viewer: bool = True
    max_fps: int = 60


@dataclass
class RobotConfig:
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


@dataclass
class ObjectConfig:
    type: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float] | Tuple[float, float, float] | Tuple[float] | None = None


@dataclass
class ControlConfig:
    joint_speed_limit: float = 0.5
    cartesian_speed_limit: float = 0.1
    default_tcp: Tuple[float, float, float, float, float, float, float] = (
        0, 0, 0, 1, 0, 0, 0
    )
    strict_cartesian: bool = True
    # --- NEW: collision control ---
    collision_check: bool = True
    ccd_substeps: int = 5
    allowed_collision_links: List[Tuple[str, str]] = field(default_factory=list)
    ground_plane_z: float = 0.0
    # Uniform shrink factor for all collision meshes (1.0 = no change)
    collision_mesh_shrink: float = 1.0
    # Auto-allow any self-collision pairs present at the home pose
    auto_allow_home_collisions: bool = True
    # Show collision meshes in the viewer for debugging (if supported by backend)
    visualize_collision_meshes: bool = False
    # --- NEW: planning control ---
    planner: str = "RRTConnect"
    planner_timeout: float = 3.0
    planner_resolution: float = 0.02
    planner_max_retry: int = 1
    cartesian_waypoints: int = 100
    postcheck_time_s: float = 0.2


@dataclass
class SimforgeConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    robots: List[RobotConfig] = field(default_factory=list)
    objects: List[ObjectConfig] = field(default_factory=list)
    control: ControlConfig = field(default_factory=ControlConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "SimforgeConfig":
        data = yaml.safe_load(Path(path).read_text())

        scene = SceneConfig(**(data.get("scene") or {}))
        robots = [RobotConfig(**r) for r in (data.get("robots") or [])]
        objects = [ObjectConfig(**o) for o in (data.get("objects") or [])]
        control = ControlConfig(**(data.get("control") or {}))
        return SimforgeConfig(scene=scene, robots=robots, objects=objects, control=control)
