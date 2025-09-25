"""Configuration schema for Simforge environments.

These models represent user-authored environment specifications. They are
pure data containers (no runtime state) and remain immutable after
validation. They deliberately avoid referencing infrastructure adapters so
that configuration can be consumed by both the runtime and tooling (e.g.,
linters, scenario generators).
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

try:  # pydantic v2
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - fallback for pydantic v1
    ConfigDict = None  # type: ignore

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


class _FrozenModel(BaseModel):
    """Base model that forbids extra fields and is immutable."""

    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=False)
    else:  # pragma: no cover - exercised only on pydantic v1
        class Config:
            extra = "forbid"
            allow_mutation = False


class Backend(str, Enum):
    """Supported Genesis backends."""

    GPU = "gpu"
    CUDA = "cuda"
    CPU = "cpu"


class ControlMode(str, Enum):
    """Control space for a robot."""

    JOINT = "joint"
    CARTESIAN = "cartesian"


class PlannerStrategy(str, Enum):
    """High-level planner selection policy."""

    CARTESIAN_PREFERRED = "cartesian_preferred"
    JOINT_ONLY = "joint_only"
    JOINT_WITH_FALLBACK = "joint_with_fallback"


class ViewerOptions(_FrozenModel):
    enabled: bool = True
    camera_position: Vec3 = (3.0, 0.0, 2.0)
    camera_look_at: Vec3 = (0.0, 0.0, 0.5)
    max_fps: int = Field(default=60, ge=1, le=240)


class SceneConfig(_FrozenModel):
    dt: float = Field(default=0.005, gt=0.0)
    gravity: Vec3 = (0.0, 0.0, -9.81)
    backend: Backend = Backend.GPU
    viewer: ViewerOptions = Field(default_factory=ViewerOptions)
    # Legacy support
    show_viewer: Optional[bool] = None
    max_fps: Optional[int] = None


class ControlLimits(_FrozenModel):
    joint_velocity: float = Field(default=1.5, gt=0.0)
    joint_acceleration: float = Field(default=3.0, gt=0.0)
    joint_jerk: Optional[float] = Field(default=None, gt=0.0)
    cartesian_velocity: float = Field(default=0.1, gt=0.0)
    cartesian_acceleration: float = Field(default=0.3, gt=0.0)


class PlannerBudgets(_FrozenModel):
    cartesian_timeout_s: float = Field(default=1.0, gt=0.0)
    joint_timeout_s: float = Field(default=3.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=20)


class RobotControlPolicy(_FrozenModel):
    mode: ControlMode = ControlMode.JOINT
    limits: ControlLimits = Field(default_factory=ControlLimits)
    planner_strategy: PlannerStrategy = PlannerStrategy.CARTESIAN_PREFERRED
    planner_budgets: PlannerBudgets = Field(default_factory=PlannerBudgets)
    ground_clearance_m: float = Field(default=0.04, ge=0.0)
    allow_self_collision: bool = False
    allow_world_collision: bool = False


class ToolAttachment(_FrozenModel):
    name: str
    tcp_offset: Optional[Tuple[float, float, float, float, float, float, float]] = None
    attach_pose: Optional[Tuple[float, float, float, float, float, float]] = None
    urdf_override: Optional[str] = None


class RobotMount(_FrozenModel):
    position: Vec3 = (0.0, 0.0, 0.0)
    orientation_rpy: Vec3 = (0.0, 0.0, 0.0)


class RobotSpec(_FrozenModel):
    """Declarative robot specification for the environment."""

    name: str
    profile: Optional[str] = Field(
        default=None,
        description="Named profile defined in presets."
    )
    urdf: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to robot URDF."
    )
    end_effector_link: Optional[str] = None
    base_pose: RobotMount = Field(default_factory=RobotMount)
    fixed_base: bool = True
    control: RobotControlPolicy = Field(default_factory=RobotControlPolicy)
    initial_joint_positions_deg: Optional[List[float]] = None
    tool: Optional[ToolAttachment] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class WorldObjectType(str, Enum):
    PLANE = "plane"
    BOX = "box"
    SPHERE = "sphere"
    URDF = "urdf"


class WorldObject(_FrozenModel):
    type: WorldObjectType
    name: str
    size: Optional[Vec3] = None
    radius: Optional[float] = Field(default=None, gt=0.0)
    pose_position: Vec3 = (0.0, 0.0, 0.0)
    pose_orientation_rpy: Vec3 = (0.0, 0.0, 0.0)
    urdf: Optional[str] = None
    dynamic: bool = False
    collision_enabled: bool = True


class WorldConfig(_FrozenModel):
    objects: List[WorldObject] = Field(default_factory=list)


class SafetyPolicy(_FrozenModel):
    enforce_workspace_bounds: bool = True
    workspace_min: Vec3 = (-2.0, -2.0, 0.0)
    workspace_max: Vec3 = (2.0, 2.0, 2.0)
    minimum_clearance_m: float = Field(default=0.01, ge=0.0)


class LoggingPolicy(_FrozenModel):
    level: str = Field(default="INFO", pattern=r"^[A-Z]+$")
    structured: bool = False
    telemetry_enabled: bool = False
    telemetry_path: Optional[str] = None


class PoliciesConfig(_FrozenModel):
    safety: SafetyPolicy = Field(default_factory=SafetyPolicy)
    logging: LoggingPolicy = Field(default_factory=LoggingPolicy)
    # Additional fields for backwards compatibility and extended policies
    ground_clearance_m: Optional[float] = Field(default=None, ge=0.0)
    planner_timeouts_s: Optional[Dict[str, float]] = Field(default=None)


class Metadata(_FrozenModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    annotations: Dict[str, str] = Field(default_factory=dict)


class EnvironmentConfig(_FrozenModel):
    """Top-level configuration document for a simulation session."""

    version: int = Field(default=2, ge=2)
    metadata: Metadata = Field(default_factory=Metadata)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    robots: List[RobotSpec] = Field(default_factory=list)
    world: WorldConfig = Field(default_factory=WorldConfig)
    policies: PoliciesConfig = Field(default_factory=PoliciesConfig)
    includes: List[str] = Field(
        default_factory=list,
        description="Additional config fragments resolved by the loader."
    )


__all__ = [
    "Backend",
    "ControlMode",
    "PlannerStrategy",
    "ViewerOptions",
    "SceneConfig",
    "ControlLimits",
    "PlannerBudgets",
    "RobotControlPolicy",
    "ToolAttachment",
    "RobotMount",
    "RobotSpec",
    "WorldObject",
    "WorldConfig",
    "SafetyPolicy",
    "LoggingPolicy",
    "PoliciesConfig",
    "Metadata",
    "EnvironmentConfig",
]
