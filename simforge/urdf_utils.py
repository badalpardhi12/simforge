from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


@dataclass
class URDFJoint:
    name: str
    joint_type: str
    lower: Optional[float]
    upper: Optional[float]


def parse_joint_limits(urdf_path: str | Path) -> List[URDFJoint]:
    path = Path(urdf_path)
    if not path.exists():
        return []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return []

    joints: List[URDFJoint] = []
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        jtype = joint.attrib.get("type", "revolute")
        limit = joint.find("limit")
        lower = upper = None
        if limit is not None:
            if "lower" in limit.attrib:
                try:
                    lower = float(limit.attrib["lower"])
                except ValueError:
                    lower = None
            if "upper" in limit.attrib:
                try:
                    upper = float(limit.attrib["upper"])
                except ValueError:
                    upper = None
        joints.append(URDFJoint(name=name, joint_type=jtype, lower=lower, upper=upper))

    # Filter to actuated joints
    joints = [j for j in joints if j.joint_type in ("revolute", "prismatic", "continuous")]
    return joints


def select_end_effector_link(urdf_path: str | Path) -> Optional[str]:
    """Select EE link as the child of the last functional (revolute/prismatic) joint
    along the longest actuated chain from the base. This better matches typical tool flange.
    """
    path = Path(urdf_path)
    if not path.exists():
        return None
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception:
        return None

    joints = []
    for j in root.findall("joint"):
        name = j.attrib.get("name", "")
        jtype = j.attrib.get("type", "")
        parent = j.find("parent").attrib.get("link") if j.find("parent") is not None else None
        child = j.find("child").attrib.get("link") if j.find("child") is not None else None
        joints.append((name, jtype, parent, child))

    # Build adjacency from parent->children for actuated joints only
    children: Dict[str, List[str]] = {}
    actuated = {c for (_n, t, _p, c) in joints if c and t in ("revolute", "prismatic", "continuous")}
    parents_map: Dict[str, str] = {c: p for (_n, _t, p, c) in joints if c and p}
    links = {p for (_n, _t, p, _c) in joints if p} | {c for (_n, _t, _p, c) in joints if c}
    base_candidates = [l for l in links if l not in parents_map]
    base = base_candidates[0] if base_candidates else None
    if not base:
        return None

    for (_n, t, p, c) in joints:
        if p and c and t in ("revolute", "prismatic", "continuous"):
            children.setdefault(p, []).append(c)

    # DFS for longest path through actuated joints
    best_path: List[str] = []
    stack = [(base, [base])]
    while stack:
        node, path_acc = stack.pop()
        next_children = children.get(node, [])
        if not next_children and len(path_acc) > len(best_path):
            best_path = path_acc
        for c in next_children:
            stack.append((c, path_acc + [c]))

    if len(best_path) >= 2:
        return best_path[-1]
    return None
