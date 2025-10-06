"""Command definitions for robot control."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
from enum import Enum, IntEnum
import uuid
import time


class CommandType(Enum):
    """Types of robot commands."""
    MOVE_JOINT = "move_joint"
    MOVE_CARTESIAN = "move_cartesian"
    SET_JOINT_POSITIONS = "set_joint_positions"
    SET_GRIPPER = "set_gripper"
    STOP = "stop"
    HOME = "home"
    TRAJECTORY = "trajectory"


class CommandPriority(IntEnum):
    """Priority levels for commands."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 10
    EMERGENCY_STOP = 11


@dataclass
class Command:
    """Base class for all commands."""
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    robot_name: str = ""
    command_type: CommandType = CommandType.STOP
    timestamp: float = field(default_factory=time.time)
    priority: CommandPriority = CommandPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare commands by priority for queue ordering."""
        if isinstance(other, Command):
            return self.priority.value < other.priority.value
        return False

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    @property
    def robot(self) -> str:
        """Alias used by legacy code paths expecting `command.robot`."""
        return self.robot_name

    @robot.setter
    def robot(self, value: str) -> None:
        self.robot_name = value


@dataclass
class MoveJointCommand(Command):
    """Command to move a single joint."""
    joint_index: int = 0
    target_position: float = 0.0
    duration: float = 1.0
    velocity_limit: Optional[float] = None
    acceleration_limit: Optional[float] = None
    
    def __post_init__(self):
        self.command_type = CommandType.MOVE_JOINT


@dataclass
class MoveCartesianCommand(Command):
    """Command for Cartesian space movement."""
    target_pose: Dict[str, Any] = field(default_factory=dict)  # position and orientation
    duration: float = 1.0
    linear_velocity_limit: Optional[float] = None
    angular_velocity_limit: Optional[float] = None
    use_collision_checking: bool = True
    reference_frame: str = "world"
    
    def __post_init__(self):
        self.command_type = CommandType.MOVE_CARTESIAN


@dataclass
class CartesianMoveCommand(Command):
    """High-level Cartesian move used by the new coordinator."""

    position_m: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    duration: float = 1.0
    reference_frame: str = "world"
    use_collision_checking: bool = True

    def __post_init__(self):
        self.command_type = CommandType.MOVE_CARTESIAN


@dataclass
class SetJointPositionsCommand(Command):
    """Command to set multiple joint positions."""
    positions: List[float] = field(default_factory=list)
    duration: float = 1.0
    synchronous: bool = True
    velocity_limits: Optional[List[float]] = None
    
    def __post_init__(self):
        self.command_type = CommandType.SET_JOINT_POSITIONS


@dataclass
class SetGripperCommand(Command):
    """Command to control gripper."""
    position: float = 0.0  # 0.0 = closed, 1.0 = open
    force: Optional[float] = None
    duration: float = 0.5
    
    def __post_init__(self):
        self.command_type = CommandType.SET_GRIPPER


@dataclass
class StopCommand(Command):
    """Emergency stop command."""
    deceleration: float = 10.0  # rad/s^2
    
    def __post_init__(self):
        self.command_type = CommandType.STOP
        self.priority = CommandPriority.EMERGENCY


@dataclass
class HomeCommand(Command):
    """Command to home the robot."""
    duration: float = 3.0
    home_position: Optional[List[float]] = None
    
    def __post_init__(self):
        self.command_type = CommandType.HOME


@dataclass
class TrajectoryCommand(Command):
    """Command to execute a full trajectory."""
    waypoints: np.ndarray = field(default_factory=lambda: np.array([]))
    times: List[float] = field(default_factory=list)
    interpolation: str = "cubic"
    
    def __post_init__(self):
        self.command_type = CommandType.TRAJECTORY


@dataclass
class JointTargetsCommand(Command):
    """Joint target command mirroring legacy behaviour."""

    values_deg: List[float] = field(default_factory=list)
    duration: float = 1.0
    synchronous: bool = True

    def __post_init__(self):
        self.command_type = CommandType.SET_JOINT_POSITIONS


__all__ = [
    "CommandType",
    "Command",
    "CommandPriority",
    "MoveJointCommand",
    "CartesianMoveCommand",
    "MoveCartesianCommand",
    "SetJointPositionsCommand",
    "SetGripperCommand",
    "StopCommand",
    "HomeCommand",
    "TrajectoryCommand",
    "JointTargetsCommand",
]
