#!/usr/bin/env python3
"""
Valid8 Robot Calibration Extraction Script

This script automates the extraction of kinematic calibration data from
a real Universal Robots arm. The calibration data is essential for accurate
kinematic calculations and trajectory planning.

The script:
1. Connects to the robot via the provided IP address
2. Extracts the factory calibration data
3. Saves it to the valid8_cell_control/config directory

Usage:
    python3 extract_calibration.py --robot-ip 192.168.1.9
    python3 extract_calibration.py --robot-ip 192.168.1.9 --output my_calibration.yaml
    
Alternatively, use the provided launch file:
    ros2 launch ur_calibration calibration_correction.launch.py \\
        robot_ip:=192.168.1.9 \\
        target_filename:=/path/to/valid8_cell_control/config/kinematics_calibration.yaml

Reference:
    https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/robot_setup.html#extract-calibration-information
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_package_path(package_name: str) -> Path:
    """Get the path to a ROS 2 package."""
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", package_name],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()) / "share" / package_name
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Package '{package_name}' not found. Is it installed?")


def extract_calibration(robot_ip: str, output_file: Path, ur_type: str = "ur5e"):
    """Extract calibration from robot using ur_calibration package.
    
    Args:
        robot_ip: IP address of the robot
        output_file: Path to save the calibration YAML
        ur_type: Type of UR robot (ur3, ur3e, ur5, ur5e, ur10, ur10e, ur16e, ur20, ur30)
    """
    print(f"[Valid8 Calibration] Extracting calibration from robot at {robot_ip}")
    print(f"[Valid8 Calibration] Output file: {output_file}")
    print(f"[Valid8 Calibration] Robot type: {ur_type}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build the launch command
    cmd = [
        "ros2", "launch", "ur_calibration", "calibration_correction.launch.py",
        f"robot_ip:={robot_ip}",
        f"target_filename:={output_file}",
    ]
    
    print(f"[Valid8 Calibration] Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print(f"[Valid8 Calibration] SUCCESS! Calibration saved to: {output_file}")
        print()
        print("[Valid8 Calibration] Next steps:")
        print("  1. Rebuild your workspace: colcon build")
        print("  2. Source the workspace: source install/setup.bash")
        print("  3. Start the robot: ros2 launch valid8_cell_control start_robot.launch.py")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Valid8 Calibration] ERROR: Calibration extraction failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure the robot is powered on and network accessible")
        print("  2. Check the robot IP address is correct")
        print("  3. Ensure ur_calibration package is installed:")
        print("     sudo apt install ros-humble-ur-calibration")
        return False


def copy_default_calibration(ur_type: str, output_file: Path):
    """Copy default calibration from ur_description package.
    
    Use this if you cannot connect to the robot or for initial testing.
    NOTE: Default calibration is not accurate for your specific robot!
    
    Args:
        ur_type: Type of UR robot
        output_file: Path to save the calibration YAML
    """
    print(f"[Valid8 Calibration] Copying default calibration for {ur_type}")
    print("[Valid8 Calibration] WARNING: Default calibration is not robot-specific!")
    
    try:
        ur_description_path = get_package_path("ur_description")
        default_calibration = ur_description_path / "config" / ur_type / "default_kinematics.yaml"
        
        if not default_calibration.exists():
            raise FileNotFoundError(f"Default calibration not found: {default_calibration}")
        
        # Copy the file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(default_calibration, output_file)
        
        print(f"[Valid8 Calibration] Copied to: {output_file}")
        print()
        print("[Valid8 Calibration] WARNING: For accurate operation, extract calibration")
        print("  from your real robot when possible.")
        return True
        
    except Exception as e:
        print(f"[Valid8 Calibration] ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract kinematic calibration from a Universal Robots arm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from real robot
  python3 extract_calibration.py --robot-ip 192.168.1.9
  
  # Extract with custom output file
  python3 extract_calibration.py --robot-ip 192.168.1.9 --output my_robot.yaml
  
  # Copy default calibration (for testing only)
  python3 extract_calibration.py --use-default --ur-type ur5e
        """
    )
    
    parser.add_argument(
        "--robot-ip",
        type=str,
        default="192.168.1.9",
        help="IP address of the robot (default: 192.168.1.9)"
    )
    
    parser.add_argument(
        "--ur-type",
        type=str,
        default="ur5e",
        choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20", "ur30"],
        help="Type of UR robot (default: ur5e)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (relative to valid8_cell_control/config or absolute path)"
    )
    
    parser.add_argument(
        "--use-default",
        action="store_true",
        help="Copy default calibration from ur_description instead of extracting from robot"
    )
    
    args = parser.parse_args()
    
    # Determine output file path
    if args.output:
        if os.path.isabs(args.output):
            output_file = Path(args.output)
        else:
            try:
                pkg_path = get_package_path("valid8_cell_control")
                output_file = pkg_path / "config" / args.output
            except RuntimeError:
                # Package not installed, use source path
                script_dir = Path(__file__).parent
                output_file = script_dir.parent / "config" / args.output
    else:
        try:
            pkg_path = get_package_path("valid8_cell_control")
            output_file = pkg_path / "config" / "kinematics_calibration.yaml"
        except RuntimeError:
            # Package not installed, use source path
            script_dir = Path(__file__).parent
            output_file = script_dir.parent / "config" / "kinematics_calibration.yaml"
    
    print()
    print("=" * 60)
    print("  Valid8 Robot Calibration Extraction")
    print("=" * 60)
    print()
    
    if args.use_default:
        success = copy_default_calibration(args.ur_type, output_file)
    else:
        success = extract_calibration(args.robot_ip, output_file, args.ur_type)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
