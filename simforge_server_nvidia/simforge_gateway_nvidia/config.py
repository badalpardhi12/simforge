"""
Robot configuration, constants, and shared dataclasses.

All modules import from here to avoid circular dependencies.
"""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Robot configuration ──────────────────────────────────────────

ROBOT_CONFIG: Dict[str, dict] = {
    "nakul_ur5e": {
        "prefix": "nakul_",
        "ip": "192.168.1.9",
        "joints": [
            "nakul_shoulder_pan_joint",
            "nakul_shoulder_lift_joint",
            "nakul_elbow_joint",
            "nakul_wrist_1_joint",
            "nakul_wrist_2_joint",
            "nakul_wrist_3_joint",
        ],
        "controller": "nakul_scaled_joint_trajectory_controller",
        "ee_link": "nakul_tool0",
        "ik_tip_link": "nakul_tool_tip_link",
        "base_link": "nakul_base_link",
        "home_position": [0.0, -math.pi / 2, 0.0,
                          -math.pi / 2, 0.0, 0.0],
        "curobo_config": "nakul_ur5e_curobo.yml",
    },
    "sahadev_ur5e": {
        "prefix": "sahadev_",
        "ip": "192.168.1.16",
        "joints": [
            "sahadev_shoulder_pan_joint",
            "sahadev_shoulder_lift_joint",
            "sahadev_elbow_joint",
            "sahadev_wrist_1_joint",
            "sahadev_wrist_2_joint",
            "sahadev_wrist_3_joint",
        ],
        "controller": "sahadev_scaled_joint_trajectory_controller",
        "ee_link": "sahadev_tool0",
        "ik_tip_link": "sahadev_tool0",
        "base_link": "sahadev_base_link",
        "home_position": [0.0, -math.pi / 2, 0.0,
                          -math.pi / 2, 0.0, 0.0],
        "curobo_config": "sahadev_ur5e_curobo.yml",
    },
}

KNOWN_OBJECTS = ["face_link", "table_link", "shop_floor"]

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
