from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from ikpy.chain import Chain
from ikpy.urdf import URDF


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
    """
    Calculates the transformation from the base of the URDF to the specified link.
    Returns a tuple of (position, quaternion_wxyz).
    """
    try:
        # Use ikpy to parse the URDF and find the transform
        temp_chain = Chain.from_urdf_file(urdf_path, last_link_vector=[0, 0, 1])
        # Find the index of the link
        link_names = [link.name for link in temp_chain.links]
        if link_name not in link_names:
            return None
        
        # Get the transformation matrix
        # Create a zero-joint configuration, as we only care about the static transform
        q_zero = np.zeros(len(temp_chain.links))
        matrix = temp_chain.forward_kinematics(q_zero, full_kinematics=True)[link_names.index(link_name)]

        pos = matrix[:3, 3]
        # Convert rotation matrix to quaternion
        # (Assuming a standard conversion, could use a library like scipy)
        # For simplicity, we'll use a basic conversion here
        from .utils import rotation_matrix_to_quat_wxyz
        quat = rotation_matrix_to_quat_wxyz(matrix[:3, :3])
        return pos, quat
    except Exception:
        return None

def merge_urdfs(robot_urdf_path: str, tool_urdf_path: str, attach_to_link: str, new_urdf_path: str):
    """
    Merges a tool URDF into a robot URDF and saves it to a new file.
    The tool is attached to the specified link with a fixed joint.
    """
    robot_tree = ET.parse(robot_urdf_path)
    robot_root = robot_tree.getroot()
    tool_tree = ET.parse(tool_urdf_path)
    tool_root = tool_tree.getroot()

    # Get the base link of the tool
    tool_base_link_element = tool_root.find("link")
    if tool_base_link_element is None:
        raise ValueError("Tool URDF must have at least one link.")
    tool_base_link = tool_base_link_element.get("name")

    # Add all links and joints from the tool to the robot
    for link in tool_root.findall("link"):
        robot_root.append(link)
    for joint in tool_root.findall("joint"):
        robot_root.append(joint)

    # Create a new fixed joint to attach the tool
    attachment_joint = ET.Element("joint", name=f"robot_tool_attachment_joint", type="fixed")
    ET.SubElement(attachment_joint, "parent", link=attach_to_link)
    ET.SubElement(attachment_joint, "child", link=tool_base_link)
    ET.SubElement(attachment_joint, "origin", xyz="0 0 0", rpy="0 0 0")
    
    robot_root.append(attachment_joint)

    # Write the merged URDF to a new file
    robot_tree.write(new_urdf_path)


def get_urdf_root_link_name(urdf_path: str | Path) -> Optional[str]:
    """
    Returns the root/base link name of a URDF by finding a link that is never a child of any joint.
    """
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
        links = {link.get("name") for link in root.findall("link")}
        children = {joint.find("child").attrib.get("link") for joint in root.findall("joint") if joint.find("child") is not None}
        bases = [l for l in links if l not in children]
        if bases:
            return bases[0]
        # Fallback to first link element
        first_link = root.find("link")
        return first_link.get("name") if first_link is not None else None
    except Exception:
        return None
