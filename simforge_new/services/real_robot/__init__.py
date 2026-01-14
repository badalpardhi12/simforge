"""Real robot driver services for controlling physical UR robots via TCP/IP.

This module provides middleware for controlling real Universal Robots (UR5e, UR10e, UR20)
using the RTDE (Real-Time Data Exchange) protocol. It supports:

- Time-parameterized trajectory execution
- Velocity-parameterized motion control
- Real-time state feedback at up to 500Hz
- Multi-robot coordination
- Seamless integration with simforge_new simulation plans

Key Components:
    URRobotConnection: Low-level RTDE connection management
    URTrajectoryExecutor: Time-parameterized trajectory execution using servoJ
    URRobotDriver: High-level driver integrating connection and execution
    URRobotRegistry: Multi-robot management and coordination

Example Usage:
    >>> from simforge_new.services.real_robot import URRobotDriver, RobotDriverConfig
    >>> config = RobotDriverConfig(robot_ip="192.168.1.9", velocity_scaling=0.5)
    >>> driver = URRobotDriver(config)
    >>> driver.connect()
    >>> driver.execute_trajectory(trajectory)
    >>> driver.disconnect()
"""

from .connection import URRobotConnection, ConnectionConfig, ConnectionState, RobotState
from .trajectory_executor import (
    URTrajectoryExecutor,
    TrajectoryExecutionConfig,
    Trajectory,
    TrajectoryPoint,
    ExecutionState,
)
from .driver import URRobotDriver, RobotDriverConfig
from .registry import URRobotRegistry

__all__ = [
    # Connection
    "URRobotConnection",
    "ConnectionConfig",
    "ConnectionState",
    "RobotState",
    # Trajectory Execution
    "URTrajectoryExecutor",
    "TrajectoryExecutionConfig",
    "Trajectory",
    "TrajectoryPoint",
    "ExecutionState",
    # Driver
    "URRobotDriver",
    "RobotDriverConfig",
    # Registry
    "URRobotRegistry",
]
