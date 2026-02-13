"""
Robot configuration, constants, and shared dataclasses.

All modules import from here to avoid circular dependencies.

The environment is selected by the ENV_CONFIG environment variable.
Default: 'valid8_dual_ur5e' (backward compatible with the original
hardcoded dual UR5e setup).

Robot IPs can be overridden via environment variables:
  ROBOT_IP_NAKUL_UR5E=192.168.1.9
  ROBOT_IP_SAHADEV_UR5E=192.168.1.16
  ROBOT_IP_UR20=10.0.0.1
"""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Environment-driven configuration ────────────────────────────

from .env_loader import load_environment, get_available_environments

ENV_CONFIG_NAME = os.environ.get("ENV_CONFIG", "valid8_dual_ur5e")

try:
    _env = load_environment(ENV_CONFIG_NAME)
    ROBOT_CONFIG: Dict[str, dict] = _env.robot_config
    KNOWN_OBJECTS: List[str] = _env.known_objects
    ENV_CONTROL_PACKAGE: str = _env.control_package
    ENV_DESCRIPTION_PACKAGE: str = _env.description_package
    ENV_WORLD_COLLISION_CONFIG: str = _env.world_collision_config
    ENV_NAME: str = _env.name
except FileNotFoundError:
    # Fallback: if running outside Docker with no config dir,
    # provide minimal defaults so imports don't break.
    import warnings
    warnings.warn(
        f"Environment config '{ENV_CONFIG_NAME}' not found. "
        f"Using empty defaults. Available: {get_available_environments()}"
    )
    ROBOT_CONFIG = {}
    KNOWN_OBJECTS = []
    ENV_CONTROL_PACKAGE = ""
    ENV_DESCRIPTION_PACKAGE = ""
    ENV_WORLD_COLLISION_CONFIG = "world_collision.yml"
    ENV_NAME = ENV_CONFIG_NAME

# ── Mode-switch signal files (shared with start_server.sh) ───
MODE_SWITCH_FILE = "/tmp/simforge_mode_switch"
CURRENT_MODE_FILE = "/tmp/simforge_current_mode"
STACK_READY_FILE = "/tmp/simforge_stack_ready"

# ── Config directory ─────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent.parent / "config"
if not CONFIG_DIR.exists():
    CONFIG_DIR = Path(
        "/ros2_ws/install/simforge_gateway_nvidia/share/"
        "simforge_gateway_nvidia/config"
    )


# ── Helper dataclasses ───────────────────────────────────────────


@dataclass
class ConnectedClient:
    """Represents a connected WebSocket client."""
    client_id: str
    websocket: Any
    connected_at: float
    last_activity: float
    heartbeat_count: int = 0


@dataclass
class RobotStateInfo:
    """Cached joint state for a single robot."""
    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: List[float] = field(default_factory=list)
    last_update: float = 0.0


# ── ur_rtde availability ─────────────────────────────────────────

try:
    import rtde_control   # noqa: F401
    import rtde_receive    # noqa: F401
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False

# ── cuRobo availability ─────────────────────────────────────────

CUROBO_AVAILABLE = False
CUROBO_IMPORT_ERROR = ""

try:
    import torch  # noqa: F401
    from curobo.types.math import Pose as CuPose  # noqa: F401
    from curobo.types.robot import JointState as CuJointState  # noqa: F401
    from curobo.types.base import TensorDeviceType  # noqa: F401
    from curobo.geom.types import WorldConfig  # noqa: F401
    from curobo.wrap.reacher.motion_gen import (  # noqa: F401
        MotionGen, MotionGenConfig, MotionGenPlanConfig,
    )
    from curobo.util.trajectory import InterpolateType  # noqa: F401
    from curobo.util_file import load_yaml  # noqa: F401

    CUROBO_AVAILABLE = True
except ImportError as e:
    CUROBO_IMPORT_ERROR = str(e)
