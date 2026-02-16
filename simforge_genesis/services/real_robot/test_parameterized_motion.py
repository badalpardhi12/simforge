#!/usr/bin/env python3
"""Test time-parameterized and velocity-parameterized trajectory execution.

Time-parameterized: Move from A to B in a fixed time, velocity adjusts automatically.
Velocity-parameterized: Move at constant velocity, time adjusts based on distance.
"""

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("test_parameterized")


def calculate_trajectory_distance(waypoints: List[Tuple[float, ...]]) -> float:
    """Calculate total joint-space distance of trajectory."""
    total = 0.0
    for i in range(1, len(waypoints)):
        prev = waypoints[i - 1]
        curr = waypoints[i]
        # Euclidean distance in joint space
        dist = math.sqrt(sum((c - p) ** 2 for c, p in zip(curr, prev)))
        total += dist
    return total


def calculate_max_joint_movement(waypoints: List[Tuple[float, ...]]) -> float:
    """Calculate max joint movement across the trajectory."""
    if len(waypoints) < 2:
        return 0.0
    start = waypoints[0]
    end = waypoints[-1]
    return max(abs(e - s) for e, s in zip(end, start))


def main():
    from simforge_genesis.services.real_robot import (
        URRobotDriver,
        RobotDriverConfig,
        Trajectory,
    )
    
    robot_ip = "192.168.1.9"
    plan_path = "/home/badal/simforge/simforge_genesis/logs/ur5e_ir_illuminator_test_plan.json"
    
    logger.info("=" * 70)
    logger.info("TIME-PARAMETERIZED vs VELOCITY-PARAMETERIZED TRAJECTORY TEST")
    logger.info("=" * 70)
    logger.info("Robot IP: %s", robot_ip)
    logger.info("Plan file: %s", plan_path)
    
    # Load plan
    with open(plan_path) as f:
        plan_data = json.load(f)
    
    # Get object data
    obj_key = list(plan_data.keys())[0]
    obj_data = plan_data[obj_key]
    
    # Collect first 3 poses
    poses = []
    for key in sorted(obj_data.keys()):
        if key.startswith("pose_") and "waypoints" in obj_data[key]:
            poses.append((key, obj_data[key]))
            if len(poses) >= 3:
                break
    
    if len(poses) < 3:
        logger.error("Need at least 3 poses, found %d", len(poses))
        return 1
    
    logger.info("Found %d poses to test", len(poses))
    for name, data in poses:
        wp_count = len(data["waypoints"])
        logger.info("  - %s: %d waypoints", name, wp_count)
    
    # Connect to robot
    config = RobotDriverConfig(
        robot_ip=robot_ip,
        robot_name="test_ur5e",
        robot_model="UR5e",
        velocity_scaling=1.0,  # Will be adjusted per test
    )
    
    with URRobotDriver(config, logger.getChild("driver")) as driver:
        if not driver.is_ready():
            logger.error("Robot not ready!")
            return 1
        
        # Get starting position
        home_q = driver.get_joint_positions(degrees=False)
        logger.info("\nCurrent position (home): %s", 
                    [f"{math.degrees(q):.1f}°" for q in home_q])
        
        # ============================================================
        # TEST 1: TIME-PARAMETERIZED EXECUTION
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("TEST 1: TIME-PARAMETERIZED EXECUTION")
        logger.info("=" * 70)
        logger.info("Each pose will execute in a FIXED TIME regardless of distance.")
        logger.info("Velocity adjusts automatically to meet the time constraint.")
        
        fixed_duration = 5.0  # seconds per pose
        logger.info("Fixed duration per pose: %.1f seconds", fixed_duration)
        
        input("\nPress ENTER to start time-parameterized test...")
        
        for i, (pose_name, pose_data) in enumerate(poses):
            waypoints = [tuple(wp) for wp in pose_data["waypoints"]]
            
            # Calculate trajectory stats
            distance = calculate_trajectory_distance(waypoints)
            max_move = calculate_max_joint_movement(waypoints)
            implied_velocity = distance / fixed_duration if fixed_duration > 0 else 0
            
            logger.info("\n--- Pose %d: %s ---", i + 1, pose_name)
            logger.info("Waypoints: %d", len(waypoints))
            logger.info("Total joint distance: %.4f rad", distance)
            logger.info("Max joint movement: %.2f° (%.4f rad)", 
                        math.degrees(max_move), max_move)
            logger.info("Fixed duration: %.1f s", fixed_duration)
            logger.info("Implied avg velocity: %.4f rad/s", implied_velocity)
            
            # Show start and end positions
            start_pos = waypoints[0]
            end_pos = waypoints[-1]
            logger.info("Start: %s", [f"{math.degrees(q):.1f}°" for q in start_pos])
            logger.info("End:   %s", [f"{math.degrees(q):.1f}°" for q in end_pos])
            
            # First move to trajectory start
            logger.info("Moving to start position...")
            driver.move_to_joints(start_pos, velocity=0.3, blocking=True)
            time.sleep(0.5)
            
            # Create time-parameterized trajectory
            trajectory = Trajectory.from_waypoints(waypoints, duration=fixed_duration)
            
            logger.info("Executing trajectory (TIME-PARAMETERIZED)...")
            start_time = time.time()
            
            success = driver.execute_trajectory(
                trajectory, 
                blocking=True, 
                use_servo=False  # Use moveJ since servoJ needs URCap
            )
            
            actual_duration = time.time() - start_time
            
            if success:
                logger.info("✅ Completed in %.2f s (target: %.1f s)", 
                            actual_duration, fixed_duration)
            else:
                logger.error("❌ Failed!")
            
            time.sleep(1.0)
        
        # Return home
        logger.info("\nReturning to home position...")
        driver.move_to_joints(home_q, velocity=0.3, blocking=True)
        time.sleep(1.0)
        
        # ============================================================
        # TEST 2: VELOCITY-PARAMETERIZED EXECUTION
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("TEST 2: VELOCITY-PARAMETERIZED EXECUTION")
        logger.info("=" * 70)
        logger.info("Each pose will execute at a FIXED VELOCITY.")
        logger.info("Duration adjusts automatically based on distance.")
        
        fixed_velocity = 0.3  # rad/s
        logger.info("Fixed joint velocity: %.2f rad/s (%.1f°/s)", 
                    fixed_velocity, math.degrees(fixed_velocity))
        
        input("\nPress ENTER to start velocity-parameterized test...")
        
        for i, (pose_name, pose_data) in enumerate(poses):
            waypoints = [tuple(wp) for wp in pose_data["waypoints"]]
            
            # Calculate trajectory stats
            distance = calculate_trajectory_distance(waypoints)
            max_move = calculate_max_joint_movement(waypoints)
            
            # Calculate duration based on velocity
            # Use max joint movement to determine time (conservative approach)
            calculated_duration = max_move / fixed_velocity if fixed_velocity > 0 else 5.0
            # Ensure minimum duration
            calculated_duration = max(2.0, calculated_duration)
            
            logger.info("\n--- Pose %d: %s ---", i + 1, pose_name)
            logger.info("Waypoints: %d", len(waypoints))
            logger.info("Total joint distance: %.4f rad", distance)
            logger.info("Max joint movement: %.2f° (%.4f rad)", 
                        math.degrees(max_move), max_move)
            logger.info("Fixed velocity: %.2f rad/s", fixed_velocity)
            logger.info("Calculated duration: %.2f s", calculated_duration)
            
            # Show start and end positions
            start_pos = waypoints[0]
            end_pos = waypoints[-1]
            logger.info("Start: %s", [f"{math.degrees(q):.1f}°" for q in start_pos])
            logger.info("End:   %s", [f"{math.degrees(q):.1f}°" for q in end_pos])
            
            # First move to trajectory start
            logger.info("Moving to start position...")
            driver.move_to_joints(start_pos, velocity=0.3, blocking=True)
            time.sleep(0.5)
            
            # Create velocity-parameterized trajectory
            # Duration is calculated from distance/velocity
            trajectory = Trajectory.from_waypoints(waypoints, duration=calculated_duration)
            
            logger.info("Executing trajectory (VELOCITY-PARAMETERIZED)...")
            start_time = time.time()
            
            success = driver.execute_trajectory(
                trajectory, 
                blocking=True, 
                use_servo=False
            )
            
            actual_duration = time.time() - start_time
            actual_velocity = max_move / actual_duration if actual_duration > 0 else 0
            
            if success:
                logger.info("✅ Completed in %.2f s (calculated: %.2f s)", 
                            actual_duration, calculated_duration)
                logger.info("   Actual avg velocity: %.2f rad/s (target: %.2f rad/s)",
                            actual_velocity, fixed_velocity)
            else:
                logger.error("❌ Failed!")
            
            time.sleep(1.0)
        
        # Return home
        logger.info("\nReturning to home position...")
        driver.move_to_joints(home_q, velocity=0.3, blocking=True)
        
        logger.info("\n" + "=" * 70)
        logger.info("TEST COMPLETE")
        logger.info("=" * 70)
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
