#!/usr/bin/env python3
"""Simple servo trajectory test with just 2 points."""

import logging
import sys
import time
import math

# Configure logging before imports
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from connection import URRobotConnection, ConnectionConfig
from trajectory_executor import (
    URTrajectoryExecutor,
    TrajectoryExecutionConfig,
    Trajectory,
    TrajectoryPoint,
    ExecutionState,
)

logger = logging.getLogger("test_servo_simple")


def main():
    robot_ip = "192.168.1.9"
    
    logger.info("=== Simple Servo Trajectory Test ===")
    logger.info("Robot IP: %s", robot_ip)
    
    # Create connection
    conn_config = ConnectionConfig(
        robot_ip=robot_ip,
    )
    
    connection = URRobotConnection(conn_config)
    
    if not connection.connect():
        logger.error("Failed to connect to robot")
        return 1
    
    logger.info("Connected to robot")
    
    # Wait for robot to be ready
    time.sleep(1.0)
    
    try:
        rtde_r = connection.receive_interface
        rtde_c = connection.control_interface
        
        if rtde_r is None or rtde_c is None:
            logger.error("Interfaces not available")
            return 1
        
        # Get current position
        current_q = rtde_r.getActualQ()
        logger.info("Current position: %s", [f"{q:.4f}" for q in current_q])
        
        # Create a simple 2-point trajectory: current -> +5 degrees on joint 6
        start_pos = tuple(current_q)
        end_pos = tuple(
            q + (math.radians(10) if i == 5 else 0)  # 10 degrees on joint 6
            for i, q in enumerate(current_q)
        )
        
        logger.info("Start: %s", [f"{q:.4f}" for q in start_pos])
        logger.info("End:   %s", [f"{q:.4f}" for q in end_pos])
        
        # Create trajectory with 2 second duration
        trajectory_duration = 2.0
        trajectory = Trajectory(
            points=[
                TrajectoryPoint(
                    positions=start_pos,
                    time_from_start=0.0,
                ),
                TrajectoryPoint(
                    positions=end_pos,
                    time_from_start=trajectory_duration,
                ),
            ],
            total_duration=trajectory_duration,
        )
        
        logger.info("Trajectory: %d points, %.2fs duration", len(trajectory.points), trajectory.total_duration)
        
        # Create executor with default config (velocity_scaling=1.0)
        exec_config = TrajectoryExecutionConfig(
            velocity_scaling=1.0,  # Full speed
            servo_frequency=500,
        )
        
        executor = URTrajectoryExecutor(connection, exec_config, logger=logger)
        
        # Execute the trajectory
        logger.info("Starting trajectory execution...")
        
        if not executor.execute(trajectory, blocking=False):
            logger.error("Failed to start trajectory")
            return 1
        
        logger.info("Trajectory started, waiting for completion...")
        
        # Wait with timeout
        if executor.wait_for_completion(timeout=10.0):
            logger.info("Trajectory completed successfully!")
        else:
            state = executor.state
            logger.error("Trajectory did not complete! State: %s", state)
            if state == ExecutionState.RUNNING:
                logger.warning("Still running, stopping...")
                executor.stop()
        
        # Check final position
        final_q = rtde_r.getActualQ()
        logger.info("Final position: %s", [f"{q:.4f}" for q in final_q])
        
        # Calculate error
        error = [abs(f - e) for f, e in zip(final_q, end_pos)]
        max_error = max(error)
        logger.info("Max position error: %.4f rad (%.2f deg)", max_error, math.degrees(max_error))
        
        # Move back to start
        logger.info("Moving back to start position...")
        rtde_c.moveJ(list(start_pos), 0.5, 0.5)
        logger.info("Done!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
        
    except Exception as e:
        logger.exception("Test failed: %s", e)
        return 1
        
    finally:
        connection.disconnect()
        logger.info("Disconnected from robot")


if __name__ == "__main__":
    sys.exit(main())
