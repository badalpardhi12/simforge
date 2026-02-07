#!/usr/bin/env python3
"""
Valid8 Cell Startup Pipeline

This script automates the startup process for the Valid8 robot cell:

1. Check if real robot is connected (ping robot IP)
2. If real robot detected:
   - Extract kinematics calibration using ur-calibration package
   - Store calibration in config directory
3. Process URDF xacro with calibration file
4. Generate SRDF collision matrix from URDF
5. Print ready status

This runs BEFORE the ROS 2 launch file to prepare all configuration.

Usage:
    python3 startup_pipeline.py --robot-ip 192.168.1.9 --ur-type ur5e
    python3 startup_pipeline.py --mock  # Skip robot detection, use defaults
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from itertools import combinations
from pathlib import Path


# ==================== Configuration ====================

# Default paths (can be overridden)
PACKAGE_ROOT = Path("/ros2_ws/src/valid8_cell_description")
CONTROL_PKG_ROOT = Path("/ros2_ws/src/valid8_cell_control")
MOVEIT_CONFIG_ROOT = Path("/ros2_ws/src/valid8_cell_moveit_config")
MOVEIT_CONFIG_INSTALLED = Path("/ros2_ws/install/valid8_cell_moveit_config/share/valid8_cell_moveit_config")
CALIBRATION_DIR = PACKAGE_ROOT / "config" / "calibration"

# Links that are part of the environment (NOT the robot)
# Collisions with these should be CHECKED, not disabled
ENVIRONMENT_LINKS = {
    'table_link', 'face_link', 'shop_floor', 'optical_table',
    'face', 'table', 'floor', 'ground', 'obstacle', 'world',
    'robot_mount'
}

# Robot link patterns (used for self-collision matrix)
ROBOT_LINK_PATTERNS = [
    'base_link', 'shoulder', 'upper_arm', 'forearm', 
    'wrist', 'tool', 'flange', 'ft_frame', 'ee_link'
]


# ==================== Utility Functions ====================

def log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def check_robot_connectivity(robot_ip: str, timeout: float = 2.0) -> bool:
    """Check if robot is reachable via TCP connection to RTDE port."""
    log(f"Checking robot connectivity at {robot_ip}...")
    
    # Try RTDE port (30004)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((robot_ip, 30004))
        sock.close()
        
        if result == 0:
            log(f"Robot is ONLINE at {robot_ip} (RTDE port 30004)")
            return True
    except Exception as e:
        log(f"Connection error: {e}", "DEBUG")
    
    log(f"Robot is OFFLINE at {robot_ip}", "WARN")
    return False


def extract_calibration(robot_ip: str, ur_type: str, output_dir: Path) -> Path:
    """
    Extract kinematics calibration from real UR robot.
    
    Uses ros2 launch ur_calibration calibration_correction.launch.py
    (the official UR method: https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_calibration/doc/usage.html)
    
    Returns path to the calibration yaml file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename - use a fixed name so it's predictable
    output_file = output_dir / f"{ur_type}_calibration.yaml"
    
    log(f"Extracting calibration from robot at {robot_ip}...")
    log(f"Output file: {output_file}")
    
    # Use the official ros2 launch command as documented by Universal Robots
    # This is more reliable than ros2 run
    cmd = [
        "ros2", "launch", "ur_calibration", "calibration_correction.launch.py",
        f"robot_ip:={robot_ip}",
        f"target_filename:={output_file}"
    ]
    
    try:
        log(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # Give more time for calibration extraction
        )
        
        if result.returncode == 0 and output_file.exists():
            log(f"Calibration extracted successfully to {output_file}", "OK")
            
            # Copy to all locations that need the calibration file
            import shutil
            
            # Copy to valid8_cell_description config
            desc_config = PACKAGE_ROOT / "config" / "kinematics_calibration.yaml"
            if desc_config.parent.exists():
                try:
                    shutil.copy(output_file, desc_config)
                    log(f"Calibration copied to {desc_config}")
                except Exception as e:
                    log(f"Could not copy to {desc_config}: {e}", "WARN")
            
            # Copy to valid8_cell_control config (may be read-only in container)
            ctrl_config = CONTROL_PKG_ROOT / "config" / "kinematics_calibration.yaml"
            if ctrl_config.parent.exists():
                try:
                    shutil.copy(output_file, ctrl_config)
                    log(f"Calibration copied to {ctrl_config}")
                except Exception as e:
                    log(f"Could not copy to {ctrl_config}: {e}", "WARN")
            
            return output_file
        else:
            log(f"Calibration extraction failed: {result.stderr}", "ERROR")
            if result.stdout:
                log(f"stdout: {result.stdout}", "DEBUG")
            log("Using default kinematics calibration", "WARN")
            return None
            
    except subprocess.TimeoutExpired:
        log("Calibration extraction timed out (60s)", "ERROR")
        return None
    except FileNotFoundError:
        log("ur_calibration package not found, using default calibration", "WARN")
        return None


def process_xacro(ur_type: str, robot_ip: str, kinematics_file: Path = None, 
                  use_mock: bool = False, reverse_ip: str = None) -> str:
    """
    Process the workcell xacro to generate URDF.
    
    Returns the URDF content as string.
    """
    xacro_file = PACKAGE_ROOT / "urdf" / "valid8_cell.urdf.xacro"
    
    if not xacro_file.exists():
        log(f"XACRO file not found: {xacro_file}", "ERROR")
        return None
    
    # Build xacro command with parameters
    cmd = [
        "xacro",
        str(xacro_file),
        f"ur_type:={ur_type}",
        f"robot_ip:={robot_ip}",
        f"use_fake_hardware:={'true' if use_mock else 'false'}",
        "headless_mode:=true"
    ]
    
    # Add reverse_ip for headless mode - robot URScript needs to connect back to this IP
    if reverse_ip:
        cmd.append(f"reverse_ip:={reverse_ip}")
    
    if kinematics_file and kinematics_file.exists():
        cmd.append(f"kinematics_parameters_file:={kinematics_file}")
    
    log(f"Processing xacro: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            log("URDF generated successfully")
            return result.stdout
        else:
            log(f"Xacro processing failed: {result.stderr}", "ERROR")
            return None
            
    except Exception as e:
        log(f"Xacro processing error: {e}", "ERROR")
        return None


# ==================== SRDF Generation ====================

def is_robot_link(link_name: str) -> bool:
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


def parse_urdf_string(urdf_content: str):
    """Parse URDF string and extract links, joints, and collision info."""
    root = ET.fromstring(urdf_content)
    
    robot_name = root.attrib.get('name', 'robot')
    
    links = {}
    for link in root.findall('.//link'):
        link_name = link.attrib.get('name')
        if link_name:
            has_collision = link.find('collision') is not None
            links[link_name] = {
                'has_collision': has_collision,
                'is_robot': is_robot_link(link_name)
            }
    
    joints = {}
    for joint in root.findall('.//joint'):
        joint_name = joint.attrib.get('name')
        parent_elem = joint.find('parent')
        child_elem = joint.find('child')
        # Skip joints without proper parent/child elements
        if joint_name and parent_elem is not None and child_elem is not None:
            parent = parent_elem.attrib.get('link')
            child = child_elem.attrib.get('link')
            if parent and child:
                joints[joint_name] = {'parent': parent, 'child': child}
    
    return robot_name, links, joints


def generate_srdf(robot_name: str, links: dict, joints: dict, tf_prefix: str = "") -> str:
    """
    Generate SRDF content with disable_collisions tags.
    
    ONLY disables collisions for:
    - Adjacent robot links (connected by joints)
    - Robot links without collision geometry
    - Robot links that can never collide due to kinematics
    
    DOES NOT disable collisions between robot and environment!
    """
    
    # Only consider robot links for self-collision matrix
    robot_links = {name: info for name, info in links.items() if info.get('is_robot', True)}
    env_links = {name: info for name, info in links.items() if not info.get('is_robot', True)}
    
    log(f"Robot links: {len(robot_links)}")
    log(f"Environment links: {len(env_links)} - {list(env_links.keys())}")
    
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
    
    # Add special disable for base_link_inertia <-> shoulder_link
    # This is the joint connection and should not be checked
    special_adjacent = [
        ('base_link_inertia', 'shoulder_link'),
        ('base_link', 'base_link_inertia'),
        ('base_link', 'base'),
    ]
    for l1, l2 in special_adjacent:
        prefix_l1 = f"{tf_prefix}{l1}" if tf_prefix and l1 in robot_links else l1
        prefix_l2 = f"{tf_prefix}{l2}" if tf_prefix and l2 in robot_links else l2
        if prefix_l1 in links and prefix_l2 in links:
            key = tuple(sorted([prefix_l1, prefix_l2]))
            all_pairs[key] = 'Adjacent'
    
    # ===== MOUNTING EXCEPTIONS =====
    # Robot base is physically mounted on the table/robot_mount, so we must
    # disable collisions between robot base links and the mounting surfaces.
    # This is necessary because the robot origin is INSIDE the table collision box.
    robot_base_links = ['base', 'base_link', 'base_link_inertia']
    mounting_env_links = ['table_link', 'robot_mount', 'shop_floor']
    
    for robot_link in robot_base_links:
        prefixed_robot = f"{tf_prefix}{robot_link}" if tf_prefix else robot_link
        if prefixed_robot not in links:
            continue
        for env_link in mounting_env_links:
            if env_link in links:
                key = tuple(sorted([prefixed_robot, env_link]))
                all_pairs[key] = 'Adjacent'  # Physically connected
                log(f"  Disabled collision: {prefixed_robot} <-> {env_link} (mounting)")
    
    # Also disable shoulder_link vs table since it's close to the base
    shoulder = f"{tf_prefix}shoulder_link" if tf_prefix else "shoulder_link"
    if shoulder in links and 'table_link' in links:
        key = tuple(sorted([shoulder, 'table_link']))
        all_pairs[key] = 'Adjacent'
        log(f"  Disabled collision: {shoulder} <-> table_link (near base)")
    
    # CRITICAL: Disable collisions between face_link and tool/end-effector links
    # The robot needs to approach the face for scanning - these are intentional proximities
    face_link = 'face_link'
    tool_ee_links = [
        f'{tf_prefix}tool_tip_link',
        f'{tf_prefix}tool_base_link', 
        f'{tf_prefix}tool0',
        f'{tf_prefix}flange',
        f'{tf_prefix}ft_frame',
        f'{tf_prefix}wrist_3_link',
        f'{tf_prefix}wrist_2_link',
    ]
    if face_link in links:
        for tool_link in tool_ee_links:
            if tool_link in links:
                key = tuple(sorted([face_link, tool_link]))
                all_pairs[key] = 'Never'  # Intentional proximity for scanning
                log(f"  Disabled collision: {face_link} <-> {tool_link} (scanning target)")
    
    # Determine the tip link based on what's available
    all_link_names = set(links.keys())
    if f'{tf_prefix}tool_tip_link' in all_link_names:
        tip_link = f'{tf_prefix}tool_tip_link'
        gripper_links = [f'{tf_prefix}tool0', f'{tf_prefix}tool_base_link', f'{tf_prefix}tool_tip_link']
        ee_parent = f'{tf_prefix}wrist_3_link'
        ee_name = 'iphone_tool'
    elif f'{tf_prefix}tool0' in all_link_names:
        tip_link = f'{tf_prefix}tool0'
        gripper_links = [f'{tf_prefix}tool0']
        ee_parent = f'{tf_prefix}wrist_3_link'
        ee_name = 'tool_tcp'
    else:
        tip_link = f'{tf_prefix}wrist_3_link'
        gripper_links = []
        ee_parent = f'{tf_prefix}wrist_3_link'
        ee_name = 'ee_tcp'
    
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!--',
        '  Auto-generated SRDF for Valid8 Cell',
        f'  Generated: {datetime.now().isoformat()}',
        '  ',
        '  This file is regenerated at startup from the URDF.',
        '  DO NOT EDIT MANUALLY - changes will be overwritten.',
        '-->',
        f'<robot name="{robot_name}">',
        '',
        '  <!-- Planning Groups -->',
        '  <group name="ur_manipulator">',
        f'    <chain base_link="{tf_prefix}base_link" tip_link="{tip_link}" />',
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
        '  <!-- Named Poses -->',
        '  <group_state name="home" group="ur_manipulator">',
        f'    <joint name="{tf_prefix}shoulder_pan_joint" value="0" />',
        f'    <joint name="{tf_prefix}shoulder_lift_joint" value="-1.57" />',
        f'    <joint name="{tf_prefix}elbow_joint" value="1.57" />',
        f'    <joint name="{tf_prefix}wrist_1_joint" value="-1.57" />',
        f'    <joint name="{tf_prefix}wrist_2_joint" value="-1.57" />',
        f'    <joint name="{tf_prefix}wrist_3_joint" value="0" />',
        '  </group_state>',
        '',
        '  <!-- Self-Collision Matrix (auto-generated from URDF) -->',
        '  <!-- Robot-environment collisions are ENABLED (not listed here) -->',
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
    
    return '\n'.join(lines)


# ==================== Main Pipeline ====================

def run_pipeline(robot_ip: str, ur_type: str, use_mock: bool = False, 
                 tf_prefix: str = "", reverse_ip: str = None) -> bool:
    """
    Run the complete startup pipeline.
    
    Args:
        robot_ip: IP address of the robot
        ur_type: UR robot type (ur3e, ur5e, ur10e, ur16e, ur20, ur30)
        use_mock: If True, use mock hardware (simulation)
        tf_prefix: TF prefix for multi-robot setups
        reverse_ip: IP address for headless mode (robot connects back to this IP)
    
    Returns True if successful, False otherwise.
    """
    log("=" * 60)
    log("Valid8 Cell Startup Pipeline")
    log("=" * 60)
    
    kinematics_file = None
    
    # Step 1: Check robot connectivity (skip if mock mode)
    if not use_mock:
        robot_online = check_robot_connectivity(robot_ip)
        
        if robot_online:
            # Step 2: Extract calibration from real robot
            kinematics_file = extract_calibration(robot_ip, ur_type, CALIBRATION_DIR)
        else:
            log("Running without real robot - using default calibration", "WARN")
    else:
        log("Mock mode - skipping robot detection")
    
    # Step 3: Process XACRO to generate URDF
    urdf_content = process_xacro(
        ur_type=ur_type,
        robot_ip=robot_ip if not use_mock else "0.0.0.0",
        kinematics_file=kinematics_file,
        use_mock=use_mock,
        reverse_ip=reverse_ip
    )
    
    if not urdf_content:
        log("Failed to generate URDF", "ERROR")
        return False
    
    # Save generated URDF for debugging
    urdf_output = PACKAGE_ROOT / "urdf" / "valid8_cell.urdf"
    with open(urdf_output, 'w') as f:
        f.write(urdf_content)
    log(f"URDF saved to: {urdf_output}")
    
    # Step 4: Generate SRDF from URDF
    log("Generating SRDF collision matrix...")
    
    try:
        robot_name, links, joints = parse_urdf_string(urdf_content)
        srdf_content = generate_srdf(robot_name, links, joints, tf_prefix)
        
        # Save SRDF to source location
        srdf_output = MOVEIT_CONFIG_ROOT / "config" / "valid8_cell.srdf"
        srdf_output.parent.mkdir(parents=True, exist_ok=True)
        with open(srdf_output, 'w') as f:
            f.write(srdf_content)
        log(f"SRDF saved to: {srdf_output}")
        
        # Also save to installed location (required for ROS launch to find it)
        srdf_installed = MOVEIT_CONFIG_INSTALLED / "config" / "valid8_cell.srdf"
        if srdf_installed.parent.exists():
            with open(srdf_installed, 'w') as f:
                f.write(srdf_content)
            log(f"SRDF also saved to: {srdf_installed}")
        else:
            log(f"Warning: Installed config path not found: {srdf_installed.parent}", "WARN")
        
    except Exception as e:
        log(f"SRDF generation failed: {e}", "ERROR")
        return False
    
    # Step 5: Print summary
    log("")
    log("=" * 60)
    log("Startup Pipeline Complete!")
    log("=" * 60)
    log(f"  Robot Type: {ur_type}")
    log(f"  Robot IP: {robot_ip}")
    log(f"  Mock Mode: {use_mock}")
    log(f"  Calibration: {kinematics_file or 'default'}")
    log(f"  URDF: {urdf_output}")
    log(f"  SRDF: {srdf_output}")
    log("")
    log("Ready to launch: ros2 launch valid8_cell_control full_stack.launch.py")
    log("")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Valid8 Cell Startup Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.9",
        help="IP address of the UR robot (default: 192.168.1.9)"
    )
    
    parser.add_argument(
        "--ur-type",
        default="ur5e",
        choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20", "ur30"],
        help="Type of UR robot (default: ur5e)"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock hardware (skip real robot detection)"
    )
    
    parser.add_argument(
        "--tf-prefix",
        default="",
        help="TF prefix for multi-robot setups (e.g., 'robot1_')"
    )
    
    parser.add_argument(
        "--reverse-ip",
        default=None,
        help="IP address for headless mode (robot connects back to this IP). "
             "If not specified, uses 0.0.0.0 which may not work with some network configurations."
    )
    
    args = parser.parse_args()
    
    success = run_pipeline(
        robot_ip=args.robot_ip,
        ur_type=args.ur_type,
        use_mock=args.mock,
        tf_prefix=args.tf_prefix,
        reverse_ip=args.reverse_ip
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
