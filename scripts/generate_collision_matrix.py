#!/usr/bin/env python3
"""
Generate MoveIt SRDF collision matrix from URDF.
Based on: https://gist.github.com/awesomebytes/18fe75b808c4c644bd3d

This generates:
1. Adjacent link pairs (links connected by joints) - should not self-collide
2. Links without collision geometry - can't collide anyway
3. Robot links that can never reach each other due to kinematic constraints

IMPORTANT: Does NOT disable collisions between robot and environment objects!
Environment collision checking (table, face, etc.) should remain ENABLED.
"""

import xml.etree.ElementTree as ET
import sys
from itertools import combinations


# Links that are part of the environment (NOT the robot)
# Collisions with these should be CHECKED, not disabled
ENVIRONMENT_LINKS = {
    'table_link', 'face_link', 'shop_floor', 'optical_table',
    'face', 'table', 'floor', 'ground', 'obstacle'
}

# Robot links (used for self-collision matrix)
ROBOT_LINK_PATTERNS = [
    'base_link', 'shoulder', 'upper_arm', 'forearm', 
    'wrist', 'tool', 'flange', 'ft_frame', 'ee_link'
]


def is_robot_link(link_name):
    """Check if a link is part of the robot (not environment)."""
    link_lower = link_name.lower()
    
    # Explicitly exclude environment links
    if link_lower in [e.lower() for e in ENVIRONMENT_LINKS]:
        return False
    
    # Check for robot link patterns
    for pattern in ROBOT_LINK_PATTERNS:
        if pattern.lower() in link_lower:
            return True
    
    # Default: assume robot link unless it looks like environment
    return link_name not in ENVIRONMENT_LINKS


def parse_urdf(urdf_file):
    """Parse URDF and extract links, joints, and collision info."""
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    robot_name = root.attrib.get('name', 'robot')
    
    links = {}
    for link in root.findall('.//link'):
        link_name = link.attrib['name']
        has_collision = link.find('collision') is not None
        links[link_name] = {
            'has_collision': has_collision,
            'is_robot': is_robot_link(link_name)
        }
    
    joints = {}
    for joint in root.findall('.//joint'):
        joint_name = joint.attrib['name']
        parent = joint.find('parent').attrib['link']
        child = joint.find('child').attrib['link']
        joints[joint_name] = {'parent': parent, 'child': child}
    
    return robot_name, links, joints


def generate_srdf(robot_name, links, joints, output_file=None):
    """Generate SRDF content with disable_collisions tags.
    
    ONLY disables collisions for:
    - Adjacent robot links (connected by joints)
    - Robot links without collision geometry
    - Robot links that can never collide due to kinematics
    
    DOES NOT disable collisions between robot and environment!
    """
    
    # Only consider robot links for self-collision matrix
    robot_links = {name: info for name, info in links.items() if info.get('is_robot', True)}
    env_links = {name: info for name, info in links.items() if not info.get('is_robot', True)}
    
    print(f"\nRobot links: {len(robot_links)}")
    print(f"Environment links: {len(env_links)} - {list(env_links.keys())}")
    print("(Environment collision checking will be ENABLED)")
    
    adjacent_pairs = [(j['parent'], j['child']) for j in joints.values()]
    no_collision_links = [name for name, info in robot_links.items() if not info['has_collision']]
    collision_links = [name for name, info in robot_links.items() if info['has_collision']]
    
    all_pairs = {}
    
    # Adjacent robot links - disable self-collision
    for l1, l2 in adjacent_pairs:
        # Only if both are robot links
        if l1 in robot_links and l2 in robot_links:
            key = tuple(sorted([l1, l2]))
            all_pairs[key] = 'Adjacent'
    
    # Robot links without collision vs each other
    for l1, l2 in combinations(no_collision_links, 2):
        key = tuple(sorted([l1, l2]))
        all_pairs[key] = 'Never'
    
    # Robot collision links vs robot no-collision links
    for l1 in collision_links:
        for l2 in no_collision_links:
            key = tuple(sorted([l1, l2]))
            all_pairs[key] = 'Never'
    
    # Determine the tip link based on what's available
    all_link_names = set(links.keys())
    if 'tool_tip_link' in all_link_names:
        tip_link = 'tool_tip_link'
        gripper_links = ['tool0', 'tool_base_link', 'tool_tip_link']
        ee_parent = 'wrist_3_link'
        ee_name = 'iphone_tool'
    elif 'tool0' in all_link_names:
        tip_link = 'tool0'
        gripper_links = ['tool0']
        ee_parent = 'wrist_3_link'
        ee_name = 'tool_tcp'
    else:
        tip_link = 'wrist_3_link'
        gripper_links = []
        ee_parent = 'wrist_3_link'
        ee_name = 'ee_tcp'
    
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<robot name="{robot_name}">',
        '',
        '  <!-- Planning Groups -->',
        '  <group name="ur_manipulator">',
        f'    <chain base_link="base_link" tip_link="{tip_link}" />',
        '  </group>',
    ]
    
    if gripper_links:
        lines.append('')
        lines.append('  <group name="gripper">')
        for link in gripper_links:
            if link in all_link_names:
                lines.append(f'    <link name="{link}" />')
        lines.append('  </group>')
        lines.append('')
        lines.append(f'  <!-- End Effector -->')
        lines.append(f'  <end_effector name="{ee_name}" parent_link="{ee_parent}" group="gripper" />')
    
    lines.extend([
        '',
        '  <!-- Self-Collision Matrix (generated from URDF) -->',
    ])
    
    by_reason = {}
    for (l1, l2), reason in sorted(all_pairs.items()):
        by_reason.setdefault(reason, []).append((l1, l2))
    
    for reason in ['Adjacent', 'Never', 'Default']:
        if reason in by_reason:
            lines.append(f'  <!-- {reason} pairs -->')
            for l1, l2 in sorted(by_reason[reason]):
                lines.append(f'  <disable_collisions link1="{l1}" link2="{l2}" reason="{reason}" />')
    
    lines.append('')
    lines.append('</robot>')
    
    content = '\n'.join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"Written SRDF to: {output_file}")
    
    return content


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_collision_matrix.py <urdf_file> [output_srdf]")
        sys.exit(1)
    
    urdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Parsing URDF: {urdf_file}")
    robot_name, links, joints = parse_urdf(urdf_file)
    
    print(f"Robot name: {robot_name}")
    print(f"Links: {len(links)} total ({sum(1 for l in links.values() if l['has_collision'])} with collision)")
    print(f"Joints: {len(joints)}")
    
    print("\nLinks with collision geometry:")
    for name, info in sorted(links.items()):
        if info['has_collision']:
            print(f"  - {name}")
    
    print("\nLinks WITHOUT collision geometry:")
    for name, info in sorted(links.items()):
        if not info['has_collision']:
            print(f"  - {name}")
    
    print()
    content = generate_srdf(robot_name, links, joints, output_file)
    
    if not output_file:
        print("\n=== Generated SRDF ===")
        print(content)
