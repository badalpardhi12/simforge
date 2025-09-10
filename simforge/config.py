from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# Core scene/viewer options
@dataclass
class SceneConfig:
    dt: float = 0.01
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    backend: str = "gpu"  # "gpu" (auto), "cpu", "cuda"
    show_viewer: bool = True
    max_fps: int = 60


# Robot descriptor (URDF + placement + optional EE and per-robot overrides)
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
    # Optional: per-robot control overrides (merged over globals by control_for)
    control: Optional["ControlConfig"] = None


# Simple environment object
@dataclass
class ObjectConfig:
    type: str                      # "plane", "box" (future: "mesh")
    name: str = ""                 # optional unique name for ACM / debugging
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees
    size: Tuple[float, float, float] | None = None                 # for box
    collision_enabled: bool = True


# Planning + collision + execution parameters
@dataclass
class ControlConfig:
    joint_speed_limit: float = 1.0
    cartesian_speed_limit: float = 0.1
    default_tcp: Tuple[float, float, float, float, float, float, float] = (0, 0, 0, 1, 0, 0, 0)
    strict_cartesian: bool = True

    # Collision control
    collision_check: bool = True
    ccd_substeps: int = 5
    allowed_collision_links: List[Tuple[str, str]] = field(default_factory=list)
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
    world_allowed_pairs: List[Tuple[str, str]] = field(default_factory=list)  # e.g., ["ur5e_1/wrist_3_link", "obj:table1"]


@dataclass
class SimforgeConfig:
    """
    Top-level configuration container for a Simforge simulation.
    Combines scene settings, robot definitions, environment objects, and control parameters.

    Attributes:
        scene: Simulation scene parameters (backend, FPS, gravity)
        robots: List of robots to load with their URDFs and configurations
        objects: List of static objects (planes, boxes) in the scene
        control: Global control settings; can be overridden per robot
    """
    scene: SceneConfig = field(default_factory=SceneConfig)
    robots: List[RobotConfig] = field(default_factory=list)
    objects: List[ObjectConfig] = field(default_factory=list)
    control: ControlConfig = field(default_factory=ControlConfig)

    # Return effective control for a robot: per-robot override merged over globals.
    def control_for(self, robot_name: str) -> ControlConfig:
        # Find the robot entry
        rob = next((r for r in self.robots if r.name == robot_name), None)
        if rob is None or rob.control is None:
            return self.control
        # Merge global (base) with per-robot override
        base = asdict(self.control)
        override = asdict(rob.control)
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

        # Scene
        scene_map = data.get("scene") or {}
        scene = SceneConfig(**scene_map)

        # Objects
        objects: List[ObjectConfig] = []
        for obj in (data.get("objects") or []):
            if not isinstance(obj, dict):
                continue
            # Accept ObjectConfig fields directly
            objects.append(ObjectConfig(**obj))

        # Global control = defaults.control overlaid by top-level control
        defaults_map = data.get("defaults") or {}
        defaults_ctrl = defaults_map.get("control") or {}
        top_ctrl = data.get("control") or {}
        gctrl = {**defaults_ctrl, **top_ctrl}
        global_ctrl = ControlConfig(**gctrl)

        # Helper to coerce a robot mapping into RobotConfig
        def build_robot(rmap_raw: Dict[str, Any]) -> RobotConfig:
            rmap = dict(rmap_raw)

            # Optional 'pose' -> base_position/base_orientation
            pose = rmap.pop("pose", None)
            if isinstance(pose, dict):
                pos = pose.get("position")
                rpy = pose.get("rpy")
                if pos is not None:
                    rmap["base_position"] = tuple(pos)
                if rpy is not None:
                    rmap["base_orientation"] = tuple(rpy)

            # Per-robot control override (kept as a ControlConfig on the robot)
            ctrl_override = None
            if "control" in rmap and isinstance(rmap["control"], dict):
                ctrl_override = ControlConfig(**rmap.pop("control"))

            # Build RobotConfig (fields must match dataclass)
            robot = RobotConfig(**rmap)
            robot.control = ctrl_override
            return robot

        # Robots list
        robots: List[RobotConfig] = []
        for entry in (data.get("robots") or []):
            # 1) String path -> include file
            if isinstance(entry, str):
                inc_path = (path.parent / entry).resolve()
                inc = yaml.safe_load(inc_path.read_text()) or {}
                # Extract first robot from included file
                inc_robots = inc.get("robots") or []
                if not inc_robots:
                    raise ValueError(f"Included file {inc_path} has no 'robots' list")
                base_map = dict(inc_robots[0])
                # Ensure required fields exist
                if "urdf" not in base_map:
                    raise ValueError(f"Included file {inc_path} robot entry must specify 'urdf'")
                # Require end_effector_link per user instruction
                if "end_effector_link" not in base_map:
                    raise ValueError(f"Included file {inc_path} robot entry must specify 'end_effector_link'")
                robots.append(build_robot(base_map))
                continue

            if isinstance(entry, dict) and "include" in entry:
                # 2) {'include': 'file.yaml', ...overrides...}
                inc_path = (path.parent / str(entry["include"])).resolve()
                inc = yaml.safe_load(inc_path.read_text()) or {}
                inc_robots = inc.get("robots") or []
                if not inc_robots:
                    raise ValueError(f"Included file {inc_path} has no 'robots' list")
                base_map = dict(inc_robots[0])

                # overlays: copy other keys from entry (except 'include')
                overlay = {k: v for k, v in entry.items() if k != "include"}
                # Support 'pose' override (handled later in build_robot)
                base_map.update(overlay)

                # Must have urdf and end_effector_link
                if "urdf" not in base_map:
                    raise ValueError(f"Included file {inc_path} robot entry must specify 'urdf'")
                if "end_effector_link" not in base_map:
                    raise ValueError(f"Include override for {inc_path} must specify 'end_effector_link' explicitly")

                robots.append(build_robot(base_map))
                continue

            if isinstance(entry, dict):
                # 3) Inline robot mapping
                # Require end_effector_link explicitly
                if "end_effector_link" not in entry:
                    raise ValueError("Inline robot entry must specify 'end_effector_link'")
                robots.append(build_robot(entry))
                continue

            raise TypeError("robots: each item must be a mapping, or a string path to a YAML (include)")

        return SimforgeConfig(scene=scene, robots=robots, objects=objects, control=global_ctrl)
