from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
# ikpy removed; keep lightweight helpers only


def parse_joint_limits(urdf_path: str | Path) -> list[dict]:
    """Extracts joint names and limits from a URDF file."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            name = joint.get("name")
            limit = joint.find("limit")
            if limit is not None:
                lower = float(limit.get("lower", -np.inf))
                upper = float(limit.get("upper", np.inf))
                joints.append({"name": name, "lower": lower, "upper": upper})
    return joints


def select_end_effector_link(urdf_path: str | Path) -> Optional[str]:
    """Heuristically selects the last link in the URDF as the end-effector."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = [link.get("name") for link in root.findall("link")]
    return links[-1] if links else None


def get_transform_to_link(urdf_path: str, link_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Deprecated: ikpy removed. Return None so caller can fall back to flange.

    In future, implement with Pinocchio if needed.
    """
    return None

def merge_urdfs(*args, **kwargs):
    """Deprecated placeholder: ikpy-based URDF merge removed."""
    raise NotImplementedError("URDF merge is not supported in this build.")


def get_urdf_root_link_name(*args, **kwargs) -> Optional[str]:
    return None
