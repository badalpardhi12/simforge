"""
Environment configuration loader.

Reads an environment YAML file and produces the ROBOT_CONFIG dict,
KNOWN_OBJECTS list, and other environment-specific settings that the
gateway node needs.

Usage:
    from simforge_gateway_nvidia.env_loader import load_environment

    env = load_environment("valid8_dual_ur5e")
    # env.robot_config  -> Dict[str, dict]
    # env.known_objects  -> List[str]
    # env.control_package -> str
    # env.world_collision_config -> str
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ── Standard joint names for UR robots ────────────────────────────
_UR_JOINT_SUFFIXES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@dataclass
class EnvironmentConfig:
    """Parsed environment configuration."""
    name: str
    description: str
    control_package: str
    description_package: str
    world_collision_config: str
    known_objects: List[str]
    robot_config: Dict[str, dict]


def _build_joint_names(prefix: str) -> List[str]:
    """Build fully-qualified joint names with prefix."""
    return [f"{prefix}{j}" for j in _UR_JOINT_SUFFIXES]


def _parse_robot(name: str, robot_dict: dict) -> dict:
    """Parse a single robot entry from the environment YAML."""
    prefix = robot_dict.get("prefix", "")
    ip = robot_dict.get("ip", "0.0.0.0")
    home = robot_dict.get("home_position", [0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0])

    return {
        "prefix": prefix,
        "ip": ip,
        "joints": _build_joint_names(prefix),
        "controller": robot_dict["controller"],
        "ee_link": robot_dict["ee_link"],
        "ik_tip_link": robot_dict.get("ik_tip_link", robot_dict["ee_link"]),
        "base_link": robot_dict["base_link"],
        "home_position": home,
        "curobo_config": robot_dict.get("curobo_config"),
    }


def _find_env_config_dir() -> Path:
    """Locate the environments/ config directory."""
    # Option 1: Relative to this source file (dev mode)
    pkg_dir = Path(__file__).parent.parent / "config" / "environments"
    if pkg_dir.exists():
        return pkg_dir

    # Option 2: Installed ROS2 package share
    installed = Path(
        "/ros2_ws/install/simforge_gateway_nvidia/share/"
        "simforge_gateway_nvidia/config/environments"
    )
    if installed.exists():
        return installed

    raise FileNotFoundError(
        f"Cannot find environments/ config directory. "
        f"Searched: {pkg_dir}, {installed}"
    )


def get_available_environments() -> List[str]:
    """List available environment config names."""
    env_dir = _find_env_config_dir()
    return sorted(
        p.stem for p in env_dir.glob("*.yaml")
    )


def load_environment(env_name: str) -> EnvironmentConfig:
    """
    Load an environment configuration by name.

    Args:
        env_name: Name of the environment (e.g. 'valid8_dual_ur5e',
                  'face_robot_ur20').  Corresponds to a YAML file in
                  config/environments/{env_name}.yaml

    Returns:
        EnvironmentConfig with robot_config dict, known_objects, etc.

    Raises:
        FileNotFoundError: If the environment YAML doesn't exist.
        ValueError: If the YAML is malformed.
    """
    env_dir = _find_env_config_dir()
    config_path = env_dir / f"{env_name}.yaml"

    if not config_path.exists():
        available = get_available_environments()
        raise FileNotFoundError(
            f"Environment config not found: {config_path}\n"
            f"Available environments: {available}"
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    env = raw.get("environment")
    if not env:
        raise ValueError(
            f"Environment config {config_path} missing 'environment' key"
        )

    # Parse robots
    robot_config = {}
    robots_dict = env.get("robots", {})
    for robot_name, robot_data in robots_dict.items():
        robot_config[robot_name] = _parse_robot(robot_name, robot_data)

    # Allow environment variable overrides for robot IPs
    # Format: ROBOT_IP_<ROBOT_NAME_UPPER>=x.x.x.x
    for robot_name in robot_config:
        env_var = f"ROBOT_IP_{robot_name.upper()}"
        ip_override = os.environ.get(env_var)
        if ip_override:
            robot_config[robot_name]["ip"] = ip_override

    return EnvironmentConfig(
        name=env.get("name", env_name),
        description=env.get("description", ""),
        control_package=env["control_package"],
        description_package=env["description_package"],
        world_collision_config=env.get("world_collision_config", "valid8_dual_ur5e/world_collision.yml"),
        known_objects=env.get("known_objects", []),
        robot_config=robot_config,
    )
