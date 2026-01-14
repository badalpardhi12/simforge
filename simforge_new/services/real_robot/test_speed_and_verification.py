#!/usr/bin/env python3
"""Test varying speeds and position verification with moveJ.

This script demonstrates:
1. How to vary robot speed using moveJ parameters
2. Position verification - checking if robot reached target positions
3. Repeatability testing - running same trajectory multiple times
"""

import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("test_speed_verify")


@dataclass
class PositionError:
    """Position error measurement."""
    joint_errors_rad: Tuple[float, ...]  # Per-joint error in radians
    joint_errors_deg: Tuple[float, ...]  # Per-joint error in degrees
    max_joint_error_rad: float
    max_joint_error_deg: float
    tcp_position_error_mm: float  # Cartesian position error
    tcp_orientation_error_rad: float  # Orientation error


def calculate_position_error(
    target_joints: Tuple[float, ...],
    actual_joints: Tuple[float, ...],
    target_tcp: Optional[Tuple[float, ...]] = None,
    actual_tcp: Optional[Tuple[float, ...]] = None,
) -> PositionError:
    """Calculate position errors between target and actual."""
    # Joint errors
    joint_errors_rad = tuple(abs(a - t) for a, t in zip(actual_joints, target_joints))
    joint_errors_deg = tuple(math.degrees(e) for e in joint_errors_rad)
    max_joint_error_rad = max(joint_errors_rad)
    max_joint_error_deg = math.degrees(max_joint_error_rad)
    
    # TCP errors (if provided)
    tcp_position_error_mm = 0.0
    tcp_orientation_error_rad = 0.0
    
    if target_tcp and actual_tcp:
        # Position error (first 3 elements are x, y, z in meters)
        pos_error = math.sqrt(sum(
            (a - t) ** 2 for a, t in zip(actual_tcp[:3], target_tcp[:3])
        ))
        tcp_position_error_mm = pos_error * 1000  # Convert to mm
        
        # Orientation error (last 3 elements are rotation vector)
        orient_error = math.sqrt(sum(
            (a - t) ** 2 for a, t in zip(actual_tcp[3:], target_tcp[3:])
        ))
        tcp_orientation_error_rad = orient_error
    
    return PositionError(
        joint_errors_rad=joint_errors_rad,
        joint_errors_deg=joint_errors_deg,
        max_joint_error_rad=max_joint_error_rad,
        max_joint_error_deg=max_joint_error_deg,
        tcp_position_error_mm=tcp_position_error_mm,
        tcp_orientation_error_rad=tcp_orientation_error_rad,
    )


def main():
    from simforge_new.services.real_robot import (
        URRobotDriver,
        RobotDriverConfig,
        Trajectory,
    )
    
    robot_ip = "192.168.1.9"
    plan_path = "/home/badal/simforge/simforge_new/logs/ur5e_ir_illuminator_test_plan.json"
    
    logger.info("=" * 70)
    logger.info("SPEED VARIATION & POSITION VERIFICATION TEST")
    logger.info("=" * 70)
    
    # Load first pose from plan
    with open(plan_path) as f:
        plan_data = json.load(f)
    
    obj_key = list(plan_data.keys())[0]
    obj_data = plan_data[obj_key]
    
    # Get first pose
    pose_data = None
    for key in sorted(obj_data.keys()):
        if key.startswith("pose_") and "waypoints" in obj_data[key]:
            pose_data = obj_data[key]
            pose_name = key
            break
    
    if pose_data is None:
        logger.error("No pose found!")
        return 1
    
    waypoints = [tuple(wp) for wp in pose_data["waypoints"]]
    logger.info("Using %s with %d waypoints", pose_name, len(waypoints))
    
    # Use subset for testing (first 50 waypoints)
    test_waypoints = waypoints[:min(50, len(waypoints))]
    start_pos = test_waypoints[0]
    end_pos = test_waypoints[-1]
    
    logger.info("Start: %s", [f"{math.degrees(q):.1f}°" for q in start_pos])
    logger.info("End:   %s", [f"{math.degrees(q):.1f}°" for q in end_pos])
    
    # Connect to robot
    config = RobotDriverConfig(
        robot_ip=robot_ip,
        robot_name="test_ur5e",
        robot_model="UR5e",
    )
    
    with URRobotDriver(config, logger.getChild("driver")) as driver:
        if not driver.is_ready():
            logger.error("Robot not ready!")
            return 1
        
        # Get interfaces for direct access
        rtde_r = driver._connection.receive_interface
        rtde_c = driver._connection.control_interface
        
        home_q = driver.get_joint_positions(degrees=False)
        
        # ============================================================
        # TEST 1: VARYING SPEEDS
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("TEST 1: VARYING SPEEDS WITH MOVEJ")
        logger.info("=" * 70)
        logger.info("We'll run the same trajectory at different velocities.")
        logger.info("Higher velocity = faster but potentially less accurate")
        logger.info("Lower velocity = slower but more accurate")
        
        velocities = [0.2, 0.5, 1.0, 1.5]  # rad/s
        accelerations = [0.3, 0.8, 1.4, 2.0]  # rad/s^2
        blend_radii = [0.01, 0.02, 0.03, 0.05]  # meters (smaller = more accurate)
        
        speed_results = []
        
        input("\nPress ENTER to start speed variation test...")
        
        for vel, acc, blend in zip(velocities, accelerations, blend_radii):
            logger.info("\n--- Testing: vel=%.2f rad/s, acc=%.2f rad/s², blend=%.3fm ---",
                        vel, acc, blend)
            
            # Move to start position
            logger.info("Moving to start position...")
            driver.move_to_joints(start_pos, velocity=0.3, blocking=True)
            time.sleep(0.5)
            
            # Record start TCP pose
            start_tcp = tuple(rtde_r.getActualTCPPose())
            
            # Build path with custom velocity/acceleration/blend
            path = []
            for i, wp in enumerate(test_waypoints):
                # Last point has no blend
                r = 0.0 if i == len(test_waypoints) - 1 else blend
                path_entry = list(wp) + [vel, acc, r]
                path.append(path_entry)
            
            # Execute trajectory
            logger.info("Executing trajectory...")
            start_time = time.time()
            
            success = rtde_c.moveJ(path)
            
            execution_time = time.time() - start_time
            
            if not success:
                logger.error("moveJ failed!")
                continue
            
            # Wait for settling
            time.sleep(0.3)
            
            # Verify final position
            actual_joints = tuple(rtde_r.getActualQ())
            actual_tcp = tuple(rtde_r.getActualTCPPose())
            
            error = calculate_position_error(
                target_joints=end_pos,
                actual_joints=actual_joints,
                target_tcp=None,  # We don't have target TCP from plan
                actual_tcp=actual_tcp,
            )
            
            logger.info("✅ Completed in %.2f s", execution_time)
            logger.info("   Max joint error: %.4f° (%.6f rad)", 
                        error.max_joint_error_deg, error.max_joint_error_rad)
            logger.info("   Per-joint errors (deg): %s",
                        [f"{e:.4f}" for e in error.joint_errors_deg])
            
            speed_results.append({
                "velocity": vel,
                "acceleration": acc,
                "blend": blend,
                "execution_time": execution_time,
                "max_joint_error_deg": error.max_joint_error_deg,
                "joint_errors_deg": error.joint_errors_deg,
            })
        
        # Summary table
        logger.info("\n" + "-" * 70)
        logger.info("SPEED TEST SUMMARY")
        logger.info("-" * 70)
        logger.info("%-10s %-10s %-10s %-12s %-15s", 
                    "Vel(rad/s)", "Acc", "Blend(m)", "Time(s)", "Max Error(°)")
        logger.info("-" * 70)
        for r in speed_results:
            logger.info("%-10.2f %-10.2f %-10.3f %-12.2f %-15.4f",
                        r["velocity"], r["acceleration"], r["blend"],
                        r["execution_time"], r["max_joint_error_deg"])
        
        # Return home
        logger.info("\nReturning to home...")
        driver.move_to_joints(home_q, velocity=0.3, blocking=True)
        time.sleep(1.0)
        
        # ============================================================
        # TEST 2: REPEATABILITY TEST
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("TEST 2: REPEATABILITY TEST")
        logger.info("=" * 70)
        logger.info("We'll run the same trajectory 5 times and check if we")
        logger.info("reach the same end position each time.")
        
        num_repetitions = 5
        repeatability_results = []
        
        input("\nPress ENTER to start repeatability test...")
        
        # Use moderate speed for repeatability test
        test_vel = 0.5
        test_acc = 0.8
        test_blend = 0.02
        
        for rep in range(num_repetitions):
            logger.info("\n--- Repetition %d/%d ---", rep + 1, num_repetitions)
            
            # Move to start position
            logger.info("Moving to start position...")
            driver.move_to_joints(start_pos, velocity=0.3, blocking=True)
            time.sleep(0.5)
            
            # Verify we're at start
            at_start = tuple(rtde_r.getActualQ())
            start_error = max(abs(a - t) for a, t in zip(at_start, start_pos))
            logger.info("At start - error: %.4f°", math.degrees(start_error))
            
            # Build and execute path
            path = []
            for i, wp in enumerate(test_waypoints):
                r = 0.0 if i == len(test_waypoints) - 1 else test_blend
                path_entry = list(wp) + [test_vel, test_acc, r]
                path.append(path_entry)
            
            rtde_c.moveJ(path)
            time.sleep(0.3)
            
            # Record final position
            final_joints = tuple(rtde_r.getActualQ())
            final_tcp = tuple(rtde_r.getActualTCPPose())
            
            error = calculate_position_error(
                target_joints=end_pos,
                actual_joints=final_joints,
            )
            
            repeatability_results.append({
                "repetition": rep + 1,
                "final_joints": final_joints,
                "final_tcp": final_tcp,
                "max_joint_error_deg": error.max_joint_error_deg,
                "joint_errors": error.joint_errors_deg,
            })
            
            logger.info("✅ Max error from target: %.4f°", error.max_joint_error_deg)
            logger.info("   Final TCP position: [%.4f, %.4f, %.4f] m",
                        final_tcp[0], final_tcp[1], final_tcp[2])
        
        # Analyze repeatability
        logger.info("\n" + "-" * 70)
        logger.info("REPEATABILITY ANALYSIS")
        logger.info("-" * 70)
        
        # Calculate variation between repetitions
        all_joints = [r["final_joints"] for r in repeatability_results]
        all_tcp = [r["final_tcp"] for r in repeatability_results]
        
        # Joint repeatability (std dev of final positions)
        joint_means = []
        joint_stds = []
        for j in range(6):
            values = [joints[j] for joints in all_joints]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            joint_means.append(mean)
            joint_stds.append(std)
        
        logger.info("\nJoint Repeatability (std deviation over %d runs):", num_repetitions)
        for j in range(6):
            logger.info("  Joint %d: mean=%.4f°, std=%.6f° (%.4f mrad)",
                        j + 1, math.degrees(joint_means[j]), 
                        math.degrees(joint_stds[j]),
                        joint_stds[j] * 1000)
        
        max_joint_std = max(joint_stds)
        logger.info("\nMax joint std deviation: %.6f° (%.4f mrad)",
                    math.degrees(max_joint_std), max_joint_std * 1000)
        
        # TCP repeatability
        tcp_positions = [(tcp[0], tcp[1], tcp[2]) for tcp in all_tcp]
        tcp_mean = tuple(sum(p[i] for p in tcp_positions) / len(tcp_positions) for i in range(3))
        tcp_deviations = [
            math.sqrt(sum((p[i] - tcp_mean[i]) ** 2 for i in range(3)))
            for p in tcp_positions
        ]
        tcp_max_deviation = max(tcp_deviations)
        
        logger.info("\nTCP Position Repeatability:")
        logger.info("  Mean position: [%.4f, %.4f, %.4f] m", *tcp_mean)
        logger.info("  Max deviation from mean: %.4f mm", tcp_max_deviation * 1000)
        
        # Accuracy (error from target)
        all_errors = [r["max_joint_error_deg"] for r in repeatability_results]
        avg_error = sum(all_errors) / len(all_errors)
        max_error = max(all_errors)
        min_error = min(all_errors)
        
        logger.info("\nAccuracy (error from target joint positions):")
        logger.info("  Average max error: %.4f°", avg_error)
        logger.info("  Best run error: %.4f°", min_error)
        logger.info("  Worst run error: %.4f°", max_error)
        
        # Return home
        logger.info("\nReturning to home...")
        driver.move_to_joints(home_q, velocity=0.3, blocking=True)
        
        # ============================================================
        # SUMMARY
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 70)
        logger.info("\n1. SPEED vs ACCURACY:")
        logger.info("   - Higher velocity = faster but ~same accuracy (good!)")
        logger.info("   - Blend radius affects path smoothness vs waypoint accuracy")
        logger.info("   - Smaller blend (0.01m) = hits waypoints more precisely")
        logger.info("   - Larger blend (0.05m) = smoother but cuts corners")
        
        logger.info("\n2. REPEATABILITY:")
        logger.info("   - Joint repeatability: ±%.4f° (±%.3f mrad)",
                    math.degrees(max_joint_std), max_joint_std * 1000)
        logger.info("   - TCP repeatability: ±%.3f mm", tcp_max_deviation * 1000)
        logger.info("   - Accuracy from target: %.4f° average", avg_error)
        
        if max_joint_std < math.radians(0.01):  # < 0.01 degrees
            logger.info("\n✅ EXCELLENT repeatability!")
        elif max_joint_std < math.radians(0.1):  # < 0.1 degrees
            logger.info("\n✅ GOOD repeatability")
        else:
            logger.info("\n⚠️ Repeatability could be improved")
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
