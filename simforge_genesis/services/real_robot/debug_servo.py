#!/usr/bin/env python3
"""Debug servo execution - verify robot actually moves."""

import logging
import math
import sys
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("debug_servo")


def main():
    try:
        import rtde_control
        import rtde_receive
    except ImportError:
        logger.error("ur_rtde not installed!")
        return 1
    
    robot_ip = "192.168.1.9"
    
    logger.info("=" * 60)
    logger.info("DEBUG SERVO EXECUTION")
    logger.info("=" * 60)
    
    # Connect
    logger.info("Connecting to %s...", robot_ip)
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    
    logger.info("Connected!")
    
    # Check robot mode
    robot_mode = rtde_r.getRobotMode()
    safety_status = rtde_r.getSafetyMode()
    logger.info("Robot mode: %d (7=running)", robot_mode)
    logger.info("Safety mode: %d (1=normal)", safety_status)
    
    if robot_mode != 7:
        logger.error("Robot not in running mode!")
        return 1
    
    # Get current position
    start_q = rtde_r.getActualQ()
    logger.info("Start position (deg): %s", [f"{math.degrees(q):.2f}" for q in start_q])
    
    # Target: move joint 6 by 15 degrees
    target_q = list(start_q)
    target_q[5] += math.radians(15)  # +15 degrees on wrist 3
    
    logger.info("Target position (deg): %s", [f"{math.degrees(q):.2f}" for q in target_q])
    
    input("\nPress ENTER to start servo motion (15° on wrist 3)...")
    
    # Servo parameters
    dt = 0.002  # 500 Hz
    lookahead = 0.1
    gain = 300
    duration = 2.0  # seconds
    
    logger.info("Starting servo loop: %.1fs duration, dt=%.3fs", duration, dt)
    
    start_time = time.time()
    loop_count = 0
    
    try:
        while True:
            loop_count += 1
            t_start = rtde_c.initPeriod()
            
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            
            # Linear interpolation
            alpha = elapsed / duration
            current_target = [
                s + alpha * (t - s) 
                for s, t in zip(start_q, target_q)
            ]
            
            # Send servo command
            rtde_c.servoJ(current_target, 0, 0, dt, lookahead, gain)
            
            # Log progress every 500 loops (1 second)
            if loop_count % 500 == 0:
                actual_q = rtde_r.getActualQ()
                j6_actual = math.degrees(actual_q[5])
                j6_target = math.degrees(current_target[5])
                logger.info(
                    "Loop %d: elapsed=%.2fs, J6 target=%.2f°, J6 actual=%.2f°",
                    loop_count, elapsed, j6_target, j6_actual
                )
            
            rtde_c.waitPeriod(t_start)
        
        logger.info("Servo loop done, stopping...")
        rtde_c.servoStop()
        time.sleep(0.5)
        
    except Exception as e:
        logger.exception("Error during servo: %s", e)
        rtde_c.servoStop()
        return 1
    
    # Check final position
    final_q = rtde_r.getActualQ()
    logger.info("Final position (deg): %s", [f"{math.degrees(q):.2f}" for q in final_q])
    
    j6_change = math.degrees(final_q[5] - start_q[5])
    logger.info("Joint 6 moved: %.2f° (expected: 15°)", j6_change)
    
    if abs(j6_change) < 1.0:
        logger.error("❌ Robot did NOT move! ServoJ commands not being executed.")
        logger.info("")
        logger.info("Possible causes:")
        logger.info("  1. Robot not in 'Remote Control' mode on teach pendant")
        logger.info("  2. Freedrive or other mode active")
        logger.info("  3. External control URCap not running")
    else:
        logger.info("✅ Robot moved successfully!")
        
        # Move back
        input("\nPress ENTER to move back to start...")
        rtde_c.moveJ(list(start_q), 0.5, 0.5)
        logger.info("Done!")
    
    rtde_c.disconnect()
    rtde_r.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
