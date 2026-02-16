"""Core data models for SimForge."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from enum import Enum

from .config_schema import (
    ControlMode,
    EnvironmentConfig as EnvironmentDocument,
    RobotControlPolicy,
    RobotSpec,
    SceneConfig,
    ToolAttachment,
    WorldConfig,
)


@dataclass
class RobotConfig:
    """Configuration for a robot."""
    name: str
    urdf_path: str
    base_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    joint_limits: List[Tuple[float, float]] = field(default_factory=list)
    velocity_limits: List[float] = field(default_factory=list)
    acceleration_limits: List[float] = field(default_factory=list)
    home_position: List[float] = field(default_factory=list)
    ee_link_name: str = "tool0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Configuration for the simulation environment."""
    robots: Dict[str, RobotConfig] = field(default_factory=dict)
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    timestep: float = 0.01
    viewer_settings: Dict[str, Any] = field(default_factory=dict)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JointState:
    """State of a single joint."""
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    effort: float = 0.0
    name: str = ""
    index: int = 0


@dataclass
class CartesianPose:
    """Cartesian pose representation."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # quaternion
    frame_id: str = "world"
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        from scipy.spatial.transform import Rotation
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(self.orientation).as_matrix()
        matrix[:3, 3] = self.position
        return matrix
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, frame_id: str = "world") -> 'CartesianPose':
        """Create from 4x4 homogeneous transformation matrix."""
        from scipy.spatial.transform import Rotation
        position = matrix[:3, 3]
        orientation = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return cls(position=position, orientation=orientation, frame_id=frame_id)


@dataclass(frozen=True)
class RobotState:
    """Robot state snapshot used by the orchestration layer."""

    name: str
    joint_positions: Tuple[float, ...]
    joint_velocities: Tuple[float, ...]
    frame: ControlMode
    timestamp_s: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    ee_pose: Optional["Pose"] = None

    @property
    def joint_count(self) -> int:
        return len(self.joint_positions)

    def as_position_array(self) -> np.ndarray:
        return np.asarray(self.joint_positions, dtype=np.float64)

    def as_velocity_array(self) -> np.ndarray:
        return np.asarray(self.joint_velocities, dtype=np.float64)


@dataclass
class Waypoint:
    """A single waypoint in a trajectory."""
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    time: float = 0.0


@dataclass
class Trajectory:
    """Robot trajectory."""
    waypoints: List[Waypoint] = field(default_factory=list)
    duration: float = 0.0
    robot_name: str = ""
    interpolation_type: str = "cubic"
    
    def sample(self, time: float) -> Optional[Waypoint]:
        """Sample the trajectory at a given time."""
        if not self.waypoints:
            return None
            
        if time <= 0:
            return self.waypoints[0]
        
        if time >= self.duration:
            return self.waypoints[-1]
        
        # Find surrounding waypoints
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i].time <= time <= self.waypoints[i + 1].time:
                # Interpolate between waypoints
                t_rel = (time - self.waypoints[i].time) / \
                       (self.waypoints[i + 1].time - self.waypoints[i].time)
                
                positions = (1 - t_rel) * self.waypoints[i].positions + \
                           t_rel * self.waypoints[i + 1].positions
                
                return Waypoint(positions=positions, time=time)
        
        return self.waypoints[-1]


class RobotType(Enum):
    """Types of robots."""
    UR5E = "ur5e"
    UR10E = "ur10e"
    UR20 = "ur20"
    MECA500 = "meca500"
    TX2_60 = "tx2_60"
    TX2_90XL = "tx2_90xl"
    CUSTOM = "custom"


@dataclass
class CollisionObject:
    """Representation of a collision object."""
    name: str
    geometry_type: str  # "box", "sphere", "cylinder", "mesh"
    dimensions: List[float] = field(default_factory=list)
    pose: CartesianPose = field(default_factory=CartesianPose)
    mesh_path: Optional[str] = None


@dataclass
class Scene:
    """Simulation scene."""
    robots: Dict[str, RobotConfig] = field(default_factory=dict)
    objects: List[CollisionObject] = field(default_factory=list)
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    timestep: float = 0.01


# ---------------------------------------------------------------------------
# Domain models used by the control stack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]


@dataclass(frozen=True)
class MotionTarget:
    pose: Optional[Pose] = None
    joints: Optional[Tuple[float, ...]] = None
    frame: str = "world"

    @property
    def is_cartesian(self) -> bool:
        return self.pose is not None

    @property
    def is_joint_space(self) -> bool:
        return self.joints is not None


@dataclass(frozen=True)
class MountPose:
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]


@dataclass(frozen=True)
class RealRobotParams:
    """Parameters for connecting to and controlling a real robot."""
    ip: str
    default_velocity: float = 0.2  # rad/s
    default_acceleration: float = 0.3  # rad/s²
    blend_radius: float = 0.03  # rad (~1.7°) - blend for smooth motion
    settling_time: float = 0.3  # seconds


@dataclass(frozen=True)
class RobotProfile:
    name: str
    urdf: str
    mount: MountPose
    fixed_base: bool
    control: RobotControlPolicy
    end_effector_link: str
    initial_joint_positions_deg: Optional[Tuple[float, ...]] = None
    tool: Optional[ToolAttachment] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    real_robot: Optional[RealRobotParams] = None

    @property
    def joint_count(self) -> int:
        if self.initial_joint_positions_deg:
            return len(self.initial_joint_positions_deg)
        if self.metadata and "dof" in self.metadata:
            try:
                return int(self.metadata["dof"])
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        return 6


@dataclass(frozen=True)
class EnvironmentSpec:
    config: EnvironmentDocument
    scene: SceneConfig
    world: WorldConfig
    robots: Tuple[RobotProfile, ...]

    @property
    def robot_names(self) -> Tuple[str, ...]:
        return tuple(robot.name for robot in self.robots)


def _rpy_to_quaternion(rpy: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    if any(abs(angle) > math.tau for angle in (roll, pitch, yaw)):
        roll, pitch, yaw = (math.radians(roll), math.radians(pitch), math.radians(yaw))
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qw, qx, qy, qz)


def build_robot_profile(spec: RobotSpec, urdf_path: str, end_effector_link: str) -> RobotProfile:
    position = tuple(float(v) for v in spec.base_pose.position)
    orientation = _rpy_to_quaternion(tuple(float(v) for v in spec.base_pose.orientation_rpy))
    mount = MountPose(position=position, orientation=orientation)
    joints = (
        tuple(float(v) for v in spec.initial_joint_positions_deg)
        if spec.initial_joint_positions_deg
        else None
    )
    metadata: Dict[str, Any] = dict(spec.metadata or {})
    if spec.profile:
        metadata.setdefault("profile", spec.profile)
    
    # Build real robot params if configured
    real_robot_params: Optional[RealRobotParams] = None
    if spec.real_robot is not None:
        real_robot_params = RealRobotParams(
            ip=spec.real_robot.ip,
            default_velocity=spec.real_robot.default_velocity,
            default_acceleration=spec.real_robot.default_acceleration,
            blend_radius=spec.real_robot.blend_radius,
            settling_time=spec.real_robot.settling_time,
        )
    
    return RobotProfile(
        name=spec.name,
        urdf=urdf_path,
        mount=mount,
        fixed_base=spec.fixed_base,
        control=spec.control,
        end_effector_link=end_effector_link or (spec.end_effector_link or ""),
        initial_joint_positions_deg=joints,
        tool=spec.tool,
        metadata=metadata,
        real_robot=real_robot_params,
    )
