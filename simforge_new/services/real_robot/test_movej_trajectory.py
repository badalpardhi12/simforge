#!/usr/bin/env python3
"""Test trajectory execution with moveJ (no URCap required)."""

import logging
import math
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("test_movej_trajectory")


def main():
    from simforge_new.services.real_robot import (
        URRobotDriver,
        RobotDriverConfig,
        Trajectory,
    )
    
    robot_ip = "192.168.1.9"
    
    logger.info("=" * 60)
    logger.info("TRAJECTORY TEST WITH MOVEJ (no URCap needed)")
    logger.info("=" * 60)
    logger.info("Robot IP: %s", robot_ip)
    
    config = RobotDriverConfig(
        robot_ip=robot_ip,
        robot_name="test_ur5e",
        robot_model="UR5e",
        velocity_scaling=0.5,
    )
    
    with URRobotDriver(config, logger.getChild("driver")) as driver:
        if not driver.is_ready():
            logger.error("Robot not ready!")
            return 1
        
        # Get current position
        current_q = driver.get_joint_positions(degrees=False)
        logger.info("Current position (deg): %s", 
                    [f"{math.degrees(q):.1f}" for q in current_q])
        
        # Create a simple trajectory: current -> +20° on wrist 3 -> back
        start_q = tuple(current_q)
        mid_q = tuple(
            q + (math.radians(20) if i == 5 else 0)
            for i, q in enumerate(current_q)
        )
        end_q = start_q
        
        waypoints = [start_q, mid_q, end_q]
        
        logger.info("")
        logger.info("Trajectory: 3 waypoints")
        logger.info("  Start: %s", [f"{math.degrees(q):.1f}°" for q in start_q])
        logger.info("  Mid:   %s", [f"{math.degrees(q):.1f}°" for q in mid_q])
        logger.info("  End:   %s", [f"{math.degrees(q):.1f}°" for q in end_q])
        logger.info("")
        logger.info("Joint 6 (wrist 3) will move: %.1f° -> %.1f° -> %.1f°",
                    math.degrees(start_q[5]), math.degrees(mid_q[5]), math.degrees(end_q[5]))
        logger.info("")
        
        # Create trajectory with 4 second duration
        trajectory = Trajectory.from_waypoints(waypoints, duration=4.0)
        
        input("Press ENTER to execute with moveJ (robot WILL move)...")
        
        logger.info("Executing trajectory with moveJ (NOT servoJ)...")
        success = driver.execute_trajectory(
            trajectory, 
            blocking=True, 
            use_servo=False  # Use moveJ instead of servoJ
        )
        
        if success:
            logger.info("✅ Trajectory completed!")
        else:
            logger.error("❌ Trajectory failed!")
            return 1
        
        final_q = driver.get_joint_positions(degrees=False)
        logger.info("Final position (deg): %s", 
                    [f"{math.degrees(q):.1f}" for q in final_q])
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
