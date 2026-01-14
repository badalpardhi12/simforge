#!/usr/bin/env python3
"""Test script for real UR robot driver connection and movement.

This script validates:
1. Connection to real UR robot via RTDE
2. Reading robot state (joint positions, TCP pose)
3. Basic movement commands
4. Time-parameterized trajectory execution from simulation plan

Usage:
    source .simforge/bin/activate
    python -m simforge_new.services.real_robot.test_driver --robot-ip 192.168.1.9

WARNING: This script moves a real robot! Ensure:
- Robot is in Remote Control mode
- Workspace is clear
- E-stop is accessible
- You understand the movements being commanded
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import List, Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("test_driver")


def test_connection(robot_ip: str, logger: logging.Logger) -> bool:
    """Test basic connection to the robot."""
    from .connection import URRobotConnection, ConnectionConfig
    
    logger.info("=" * 60)
    logger.info("TEST 1: Connection")
    logger.info("=" * 60)
    
    config = ConnectionConfig(
        robot_ip=robot_ip,
        rtde_frequency=500.0,
    )
    
    connection = URRobotConnection(config, logger.getChild("conn"))
    
    logger.info("Attempting connection to %s...", robot_ip)
    
    success = connection.connect()
    
    if not success:
        logger.error("❌ Connection FAILED")
        return False
    
    logger.info("✅ Connection successful!")
    
    # Read initial state
    state = connection.get_state()
    
    logger.info("Robot State:")
    logger.info("  Robot mode: %d", state.robot_mode)
    logger.info("  Safety status: %d", state.safety_status)
    logger.info("  Emergency stopped: %s", state.emergency_stopped)
    logger.info("  Protective stopped: %s", state.protective_stopped)
    
    logger.info("Joint Positions (deg):")
    for i, q in enumerate(state.actual_q):
        logger.info("  Joint %d: %.2f°", i + 1, math.degrees(q))
    
    logger.info("TCP Pose:")
    logger.info("  Position: [%.4f, %.4f, %.4f] m", *state.actual_tcp_pose[:3])
    logger.info("  Rotation: [%.4f, %.4f, %.4f] rad", *state.actual_tcp_pose[3:])
    
    connection.disconnect()
    logger.info("Disconnected successfully")
    
    return True


def test_driver_basic(robot_ip: str, logger: logging.Logger) -> bool:
    """Test high-level driver functionality."""
    from .driver import URRobotDriver, RobotDriverConfig
    
    logger.info("=" * 60)
    logger.info("TEST 2: Driver Basic Operations")
    logger.info("=" * 60)
    
    config = RobotDriverConfig(
        robot_ip=robot_ip,
        robot_name="test_ur5e",
        robot_model="UR5e",
        max_joint_velocity=0.5,  # Slow for safety
        max_joint_acceleration=0.5,
        velocity_scaling=0.3,  # 30% speed
    )
    
    driver = URRobotDriver(config, logger.getChild("driver"))
    
    if not driver.connect():
        logger.error("❌ Driver connection FAILED")
        return False
    
    logger.info("✅ Driver connected!")
    
    # Check readiness
    if driver.is_ready():
        logger.info("✅ Robot is ready for motion")
    else:
        logger.warning("⚠️ Robot NOT ready for motion (check mode/safety)")
        driver.disconnect()
        return False
    
    # Get current positions
    current_joints = driver.get_joint_positions(degrees=True)
    logger.info("Current joint positions (deg): %s", 
                [f"{q:.1f}" for q in current_joints])
    
    current_tcp = driver.get_tcp_pose()
    logger.info("Current TCP pose: [%.3f, %.3f, %.3f, %.2f, %.2f, %.2f]",
                *current_tcp)
    
    driver.disconnect()
    return True


def test_small_movement(robot_ip: str, logger: logging.Logger, skip: bool = False) -> bool:
    """Test a small joint movement."""
    from .driver import URRobotDriver, RobotDriverConfig
    
    logger.info("=" * 60)
    logger.info("TEST 3: Small Movement")
    logger.info("=" * 60)
    
    if skip:
        logger.info("⏭️ Skipping movement test (--no-move flag)")
        return True
    
    config = RobotDriverConfig(
        robot_ip=robot_ip,
        robot_name="test_ur5e",
        robot_model="UR5e",
        max_joint_velocity=0.3,  # Very slow
        max_joint_acceleration=0.3,
        velocity_scaling=0.2,  # 20% speed
    )
    
    with URRobotDriver(config, logger.getChild("driver")) as driver:
        if not driver.is_ready():
            logger.error("❌ Robot not ready")
            return False
        
        current_joints = driver.get_joint_positions(degrees=False)
        
        # Create small movement: rotate joint 6 by 10 degrees
        target_joints = list(current_joints)
        target_joints[5] += math.radians(10)  # +10 degrees on wrist 3
        
        logger.info("Moving wrist 3 by +10 degrees...")
        logger.info("Current: %.2f° -> Target: %.2f°",
                    math.degrees(current_joints[5]),
                    math.degrees(target_joints[5]))
        
        input("Press Enter to execute movement (Ctrl+C to abort)...")
        
        success = driver.move_to_joints(target_joints, blocking=True)
        
        if success:
            logger.info("✅ Movement completed!")
            
            # Move back
            logger.info("Moving back to original position...")
            driver.move_to_joints(current_joints, blocking=True)
            logger.info("✅ Returned to original position")
        else:
            logger.error("❌ Movement FAILED")
            return False
    
    return True


def test_trajectory_execution(
    robot_ip: str,
    plan_path: str,
    logger: logging.Logger,
    skip: bool = False,
) -> bool:
    """Test trajectory execution from a simulation plan."""
    from .driver import URRobotDriver, RobotDriverConfig
    from .trajectory_executor import Trajectory
    
    logger.info("=" * 60)
    logger.info("TEST 4: Trajectory Execution")
    logger.info("=" * 60)
    
    if skip:
        logger.info("⏭️ Skipping trajectory test (--no-move flag)")
        return True
    
    plan_file = Path(plan_path)
    if not plan_file.exists():
        logger.error("❌ Plan file not found: %s", plan_path)
        return False
    
    # Load plan
    with open(plan_file) as f:
        plan_data = json.load(f)
    
    # Get first pose trajectory
    object_key = list(plan_data.keys())[0]
    object_data = plan_data[object_key]
    
    # Find first pose with waypoints
    pose_key = None
    pose_data = None
    for key, value in object_data.items():
        if key.startswith("pose_") and isinstance(value, dict) and "waypoints" in value:
            pose_key = key
            pose_data = value
            break
    
    if pose_data is None:
        logger.error("❌ No valid pose found in plan")
        return False
    
    waypoints = pose_data["waypoints"]
    logger.info("Found %s/%s with %d waypoints", object_key, pose_key, len(waypoints))
    
    # Use only first 10 waypoints for testing
    test_waypoints = waypoints[:min(10, len(waypoints))]
    logger.info("Using first %d waypoints for test", len(test_waypoints))
    
    config = RobotDriverConfig(
        robot_ip=robot_ip,
        robot_name="test_ur5e",
        robot_model="UR5e",
        max_joint_velocity=0.5,
        max_joint_acceleration=0.5,
        velocity_scaling=0.25,  # 25% speed for safety
    )
    
    with URRobotDriver(config, logger.getChild("driver")) as driver:
        if not driver.is_ready():
            logger.error("❌ Robot not ready")
            return False
        
        current_joints = driver.get_joint_positions(degrees=False)
        
        # First, move to start position
        start_position = test_waypoints[0]
        
        logger.info("Moving to trajectory start position...")
        logger.info("Start: %s", [f"{math.degrees(q):.1f}°" for q in start_position])
        
        input("Press Enter to move to start position (Ctrl+C to abort)...")
        
        success = driver.move_to_joints(start_position, blocking=True)
        if not success:
            logger.error("❌ Failed to reach start position")
            return False
        
        logger.info("✅ At start position")
        
        # Create trajectory
        trajectory = Trajectory.from_waypoints(test_waypoints, duration=5.0)
        
        logger.info("Executing trajectory: %d points, %.2fs duration",
                    len(trajectory.points), trajectory.total_duration)
        
        input("Press Enter to execute trajectory (Ctrl+C to abort)...")
        
        success = driver.execute_trajectory(trajectory, blocking=True, use_servo=True)
        
        if success:
            logger.info("✅ Trajectory executed successfully!")
        else:
            logger.error("❌ Trajectory execution FAILED")
        
        # Return to original position
        logger.info("Returning to original position...")
        driver.move_to_joints(current_joints, blocking=True)
        logger.info("✅ Returned to original position")
    
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test UR robot driver connection and movement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.1.9",
        help="IP address of the UR robot (default: 192.168.1.9)",
    )
    parser.add_argument(
        "--plan-file",
        default="/home/badal/simforge/simforge_new/logs/ur5e_ir_illuminator_test_plan.json",
        help="Path to simulation plan JSON for trajectory test",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="Skip tests that involve robot movement",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--test",
        choices=["connection", "driver", "movement", "trajectory", "all"],
        default="all",
        help="Specific test to run",
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.log_level)
    
    logger.info("UR Robot Driver Test Suite")
    logger.info("Robot IP: %s", args.robot_ip)
    logger.info("-" * 60)
    
    # Check that we're not accidentally running on sim IP
    if args.robot_ip.startswith("127.") or args.robot_ip == "localhost":
        logger.warning("⚠️ Using localhost/127.x.x.x - ensure URSim is running!")
    
    tests_passed = 0
    tests_failed = 0
    
    test_map = {
        "connection": lambda: test_connection(args.robot_ip, logger),
        "driver": lambda: test_driver_basic(args.robot_ip, logger),
        "movement": lambda: test_small_movement(args.robot_ip, logger, args.no_move),
        "trajectory": lambda: test_trajectory_execution(
            args.robot_ip, args.plan_file, logger, args.no_move
        ),
    }
    
    if args.test == "all":
        tests_to_run = list(test_map.keys())
    else:
        tests_to_run = [args.test]
    
    for test_name in tests_to_run:
        try:
            if test_map[test_name]():
                tests_passed += 1
            else:
                tests_failed += 1
        except KeyboardInterrupt:
            logger.info("Test aborted by user")
            break
        except Exception as exc:
            logger.exception("Test '%s' raised exception: %s", test_name, exc)
            tests_failed += 1
    
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info("✅ Passed: %d", tests_passed)
    logger.info("❌ Failed: %d", tests_failed)
    
    return 0 if tests_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
