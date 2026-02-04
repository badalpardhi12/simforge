#!/usr/bin/env python3
"""
Generate MoveIt SRDF collision matrix from URDF.

This script generates disable_collisions entries for:
1. Adjacent links in the kinematic chain
2. Links without collision geometry
3. Standard UR5e self-collision pairs (from official MoveIt configs)

Author: Based on awesomebytes' urdf_to_disable_colisions.py
"""
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple

# Official UR5e disable_collisions from MoveIt Setup Assistant
# These are the pairs that are ALWAYS in collision or NEVER in collision
# Generated with 100,000 samples
UR5E_STANDARD_COLLISIONS = [
    # Adjacent links
    ("base_link", "shoulder_link", "Adjacent"),
    ("shoulder_link", "upper_arm_link", "Adjacent"),
    ("upper_arm_link", "forearm_link", "Adjacent"),
    ("forearm_link", "wrist_1_link", "Adjacent"),
    ("wrist_1_link", "wrist_2_link", "Adjacent"),
    ("wrist_2_link", "wrist_3_link", "Adjacent"),
    
    # Never in collision (far apart)
    ("base_link", "wrist_1_link", "Never"),
    ("base_link", "wrist_2_link", "Never"),
    ("base_link", "wrist_3_link", "Never"),
    ("shoulder_link", "wrist_1_link", "Never"),
    ("shoulder_link", "wrist_2_link", "Never"),
    ("shoulder_link", "wrist_3_link", "Never"),
    ("shoulder_link", "forearm_link", "Never"),
    ("upper_arm_link", "wrist_1_link", "Never"),
    ("upper_arm_link", "wrist_2_link", "Never"),
    ("upper_arm_link", "wrist_3_link", "Never"),
    ("forearm_link", "wrist_3_link", "Never"),
    
    # Base link inertia (always coincident with base_link)
    ("base_link", "base_link_inertia", "Default"),
    ("base_link_inertia", "shoulder_link", "Adjacent"),
]


def parse_urdf(urdf_path: str) -> Tuple[Dict, Dict, Dict]:
    """Parse URDF and extract link, joint, and parent-child relationships."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Get all links
    links = {}
    for link in root.findall('.//link'):
        name = link.get('name')
        has_collision = link.find('collision') is not None
        links[name] = {'name': name, 'has_collision': has_collision}
    
    # Get all joints and build parent-child map
    joints = {}
    child_map = {}  # parent_link -> [(joint_name, child_link), ...]
    parent_map = {}  # child_link -> parent_link
    
    for joint in root.findall('.//joint'):
        name = joint.get('name')
        joint_type = joint.get('type')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        
        joints[name] = {
            'name': name,
            'type': joint_type,
            'parent': parent,
            'child': child
        }
        
        if parent not in child_map:
            child_map[parent] = []
        child_map[parent].append((name, child))
        parent_map[child] = parent
    
    return links, joints, child_map


def get_adjacent_pairs(child_map: Dict, root_link: str) -> List[Tuple[str, str]]:
    """Get all adjacent link pairs from the kinematic tree."""
    pairs = []
    
    def traverse(parent_link):
        if parent_link not in child_map:
            return
        for joint_name, child_link in child_map[parent_link]:
            pairs.append((parent_link, child_link))
            traverse(child_link)
    
    traverse(root_link)
    return pairs


def get_root_link(links: Dict, child_map: Dict) -> str:
    """Find the root link of the URDF."""
    all_links = set(links.keys())
    child_links = set()
    for parent, children in child_map.items():
        for _, child in children:
            child_links.add(child)
    
    roots = all_links - child_links
    # Prefer 'world' or 'base_link' if present
    for preferred in ['world', 'base_link']:
        if preferred in roots:
            return preferred
    return list(roots)[0] if roots else list(links.keys())[0]


def get_links_without_collision(links: Dict) -> List[str]:
    """Get links that don't have collision geometry."""
    return [name for name, data in links.items() if not data['has_collision']]


# Robot links that should be checked for self-collision
ROBOT_COLLISION_LINKS = {
    'base_link', 'base_link_inertia', 'shoulder_link', 'upper_arm_link',
    'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link'
}

# Links that have no collision mesh (virtual frames)
VIRTUAL_FRAME_LINKS = {'world', 'base', 'ft_frame', 'flange', 'tool0', 'tool_tip_link'}


def generate_disable_collisions(
    links: Dict,
    child_map: Dict,
    include_standard_ur5e: bool = True,
    tool_links: List[str] = None,
    environment_links: List[str] = None
) -> str:
    """Generate disable_collisions entries for SRDF.
    
    IMPORTANT: We ONLY disable collisions for:
    1. Adjacent links in kinematic chain (they always overlap slightly)
    2. Links without collision geometry (can't collide anyway)
    3. Fixed environment objects with each other (static, can't collide)
    4. Base link with table (robot is mounted on table)
    
    We DO NOT disable:
    - Robot arm links with tool (tool could hit arm in certain configs)
    - Robot links with environment (this is the collision checking we want!)
    """
    
    root_link = get_root_link(links, child_map)
    
    # Track which pairs we've already added
    added_pairs: Set[Tuple[str, str]] = set()
    
    def add_pair(link1: str, link2: str, reason: str) -> str:
        # Normalize order
        pair = tuple(sorted([link1, link2]))
        if pair in added_pairs:
            return ""
        added_pairs.add(pair)
        return f'  <disable_collisions link1="{pair[0]}" link2="{pair[1]}" reason="{reason}"/>\n'
    
    result = ""
    env_links_set = set(environment_links or [])
    tool_links_set = set(tool_links or [])
    
    # 1. Adjacent links from URDF traversal (only robot kinematic chain)
    result += "\n  <!-- Adjacent links (from URDF kinematic chain) -->\n"
    adjacent_pairs = get_adjacent_pairs(child_map, root_link)
    for parent, child in adjacent_pairs:
        # Skip environment adjacencies (world->table, world->face, etc.)
        # We want those checked! (unless they're both environment)
        if parent in env_links_set and child not in env_links_set:
            continue
        if child in env_links_set and parent not in env_links_set:
            continue
        result += add_pair(parent, child, "Adjacent")
    
    # 2. Links without collision mesh (virtual frames)
    no_collision_links = get_links_without_collision(links)
    if no_collision_links:
        result += "\n  <!-- Links without collision geometry (virtual frames) -->\n"
        # Disable collisions between all no-collision links
        for i, link1 in enumerate(no_collision_links):
            for link2 in no_collision_links[i+1:]:
                result += add_pair(link1, link2, "Never")
        
        # Disable collisions between no-collision links and all other links
        # (they can't collide anyway since they have no geometry)
        for link1 in no_collision_links:
            for link2 in links.keys():
                if link2 not in no_collision_links:
                    result += add_pair(link1, link2, "Never")
    
    # 3. Standard UR5e self-collision pairs (from MoveIt Setup Assistant)
    if include_standard_ur5e:
        result += "\n  <!-- Standard UR5e self-collision pairs -->\n"
        for link1, link2, reason in UR5E_STANDARD_COLLISIONS:
            # Only add if both links exist in URDF
            if link1 in links and link2 in links:
                result += add_pair(link1, link2, reason)
    
    # 4. Tool mount adjacency (only the direct connection)
    if tool_links:
        result += "\n  <!-- Tool mount adjacency -->\n"
        # Tool links are adjacent to each other
        for i, link in enumerate(tool_links[:-1]):
            if link in links and tool_links[i+1] in links:
                result += add_pair(link, tool_links[i+1], "Adjacent")
        
        # First tool link is adjacent to wrist_3_link
        if 'wrist_3_link' in links and tool_links[0] in links:
            result += add_pair('wrist_3_link', tool_links[0], "Adjacent")
    
    # 5. Environment objects - only disable between static objects
    if environment_links:
        result += "\n  <!-- Environment objects (fixed, don't collide with each other) -->\n"
        for i, link1 in enumerate(environment_links):
            for link2 in environment_links[i+1:]:
                if link1 in links and link2 in links:
                    result += add_pair(link1, link2, "Never")
        
        # Also disable base_link collision with table (robot is mounted on table)
        if 'table_link' in links and 'base_link' in links:
            result += add_pair('base_link', 'table_link', "Default")
        if 'table_link' in links and 'base_link_inertia' in links:
            result += add_pair('base_link_inertia', 'table_link', "Default")
    
    return result


def generate_full_srdf(
    urdf_path: str,
    robot_name: str = "valid8_environment",
    tool_links: List[str] = None,
    environment_links: List[str] = None
) -> str:
    """Generate complete SRDF content."""
    
    links, joints, child_map = parse_urdf(urdf_path)
    
    srdf = f'''<?xml version="1.0" encoding="UTF-8"?>
<!--
  MoveIt SRDF for {robot_name}
  Generated by generate_srdf_collisions.py
  
  This file defines:
  - Planning groups (kinematic chains)
  - End effector definitions  
  - Self-collision matrix
  - Named poses
-->
<robot name="{robot_name}">
  <!-- Planning group for the arm (base to tool0) -->
  <group name="ur_manipulator">
    <chain base_link="base_link" tip_link="tool0"/>
  </group>
  
  <!-- End effector (tool) planning group -->
  <group name="gripper">
    <link name="tool0"/>
'''
    
    # Add tool links to gripper group if specified
    if tool_links:
        for link in tool_links:
            if link in links:
                srdf += f'    <link name="{link}"/>\n'
    
    srdf += '''  </group>
  
  <!-- Virtual joint connecting world to robot base -->
  <virtual_joint name="virtual_joint" type="fixed" parent_frame="world" child_link="base_link"/>
  
  <!-- End effector definition -->
  <end_effector name="tool_tcp" parent_link="tool0" group="gripper"/>
  
  <!-- Named poses -->
  <group_state name="home" group="ur_manipulator">
    <joint name="shoulder_pan_joint" value="0.0"/>
    <joint name="shoulder_lift_joint" value="-1.5708"/>
    <joint name="elbow_joint" value="0.0"/>
    <joint name="wrist_1_joint" value="-1.5708"/>
    <joint name="wrist_2_joint" value="0.0"/>
    <joint name="wrist_3_joint" value="0.0"/>
  </group_state>
  
  <group_state name="ready" group="ur_manipulator">
    <joint name="shoulder_pan_joint" value="0.0"/>
    <joint name="shoulder_lift_joint" value="-1.5708"/>
    <joint name="elbow_joint" value="1.5708"/>
    <joint name="wrist_1_joint" value="-1.5708"/>
    <joint name="wrist_2_joint" value="-1.5708"/>
    <joint name="wrist_3_joint" value="0.0"/>
  </group_state>
  
  <!-- ========== SELF-COLLISION MATRIX ========== -->
  <!-- 
    This matrix disables collision checking for link pairs that:
    1. Are adjacent in the kinematic chain
    2. Have no collision geometry
    3. Are known to never collide based on robot geometry
    
    IMPORTANT: Environment objects (table, face) are NOT disabled here
    so that the robot will properly check for collisions with them!
  -->
'''
    
    # Generate collision matrix
    collision_matrix = generate_disable_collisions(
        links, child_map,
        include_standard_ur5e=True,
        tool_links=tool_links,
        environment_links=environment_links
    )
    srdf += collision_matrix
    
    srdf += '''
</robot>
'''
    return srdf


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SRDF collision matrix from URDF')
    parser.add_argument('urdf', help='Path to URDF file')
    parser.add_argument('--output', '-o', help='Output SRDF file path')
    parser.add_argument('--robot-name', default='valid8_environment', help='Robot name in SRDF')
    parser.add_argument('--tool-links', nargs='+', default=['tool_base_link', 'tool_tip_link'],
                       help='Tool link names')
    parser.add_argument('--env-links', nargs='+', default=['shop_floor', 'table_link', 'face_link'],
                       help='Environment link names (fixed objects)')
    
    args = parser.parse_args()
    
    srdf_content = generate_full_srdf(
        args.urdf,
        robot_name=args.robot_name,
        tool_links=args.tool_links,
        environment_links=args.env_links
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(srdf_content)
        print(f"Generated SRDF written to {args.output}")
    else:
        print(srdf_content)
