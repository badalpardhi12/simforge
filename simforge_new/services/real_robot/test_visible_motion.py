#!/usr/bin/env python3
"""Test trajectory execution with visible motion."""

import logging
import math
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("test_visible_motion")


def main():
    from simforge_new.services.real_robot import (
        URRobotDriver,
        RobotDriverConfig,
        Trajectory,
        TrajectoryPoint,
    )
    
    robot_ip = "192.168.1.9"
    
    logger.info("=" * 60)
    logger.info("VISIBLE MOTION TRAJECTORY TEST")
    logger.info("=" * 60)
    logger.info("Robot IP: %s", robot_ip)
    
    config = RobotDriverConfig(
        robot_ip=robot_ip,
        robot_name="test_ur5e",
        robot_model="UR5e",
        velocity_scaling=0.5,  # 50% speed
    )
    
    with URRobotDriver(config, logger.getChild("driver")) as driver:
        if not driver.is_ready():
            logger.error("Robot not ready!")
            return 1
        
        # Get current position
        current_q = driver.get_joint_positions(degrees=False)
        logger.info("Current position (deg): %s", 
                    [f"{math.degrees(q):.1f}" for q in current_q])
        
        # Create a trajectory that moves joint 3 (elbow) and joint 4 (wrist 1)
        # by 20 degrees over 4 seconds
        start_q = list(current_q)
        
        # Define waypoints with meaningful motion
        # We'll move joints 3 and 4 in a coordinated way
        waypoints = []
        num_points = 20
        duration = 4.0  # seconds
        
        amplitude_j3 = math.radians(15)  # 15 degrees movement on elbow
        amplitude_j4 = math.radians(20)  # 20 degrees movement on wrist 1
        
        for i in range(num_points + 1):
            t = i / num_points  # 0 to 1
            
            # Sinusoidal motion
            offset_j3 = amplitude_j3 * math.sin(math.pi * t)  # 0 -> peak -> 0
            offset_j4 = amplitude_j4 * math.sin(math.pi * t)  # 0 -> peak -> 0
            
            wp = list(start_q)
            wp[2] += offset_j3  # Joint 3 (elbow)
            wp[3] += offset_j4  # Joint 4 (wrist 1)
            
            waypoints.append(tuple(wp))
        
        # Create trajectory
        trajectory = Trajectory.from_waypoints(waypoints, duration=duration)
        
        logger.info("")
        logger.info("Trajectory details:")
        logger.info("  - %d waypoints over %.1f seconds", len(waypoints), duration)
        logger.info("  - Joint 3 (elbow) moves: 0 -> +%.1f° -> 0", math.degrees(amplitude_j3))
        logger.info("  - Joint 4 (wrist1) moves: 0 -> +%.1f° -> 0", math.degrees(amplitude_j4))
        logger.info("  - Velocity scaling: 50%% (actual execution time: %.1fs)", duration / 0.5)
        logger.info("")
        
        # Show start and peak positions
        peak_idx = num_points // 2
        logger.info("Start position (deg): %s", 
                    [f"{math.degrees(q):.1f}" for q in waypoints[0]])
        logger.info("Peak position (deg):  %s", 
                    [f"{math.degrees(q):.1f}" for q in waypoints[peak_idx]])
        logger.info("End position (deg):   %s", 
                    [f"{math.degrees(q):.1f}" for q in waypoints[-1]])
        logger.info("")
        
        input("Press ENTER to execute trajectory (the robot WILL MOVE)...")
        
        logger.info("Executing trajectory...")
        success = driver.execute_trajectory(trajectory, blocking=True)
        
        if success:
            logger.info("✅ Trajectory completed!")
        else:
            logger.error("❌ Trajectory failed!")
            return 1
        
        # Verify final position
        final_q = driver.get_joint_positions(degrees=False)
        logger.info("Final position (deg): %s", 
                    [f"{math.degrees(q):.1f}" for q in final_q])
        
        # Check error
        max_error = max(abs(f - s) for f, s in zip(final_q, start_q))
        logger.info("Max error from start: %.2f°", math.degrees(max_error))
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
