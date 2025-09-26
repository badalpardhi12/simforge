#!/usr/bin/env python3
"""
Demo script for face_robot configuration.
Moves TX2-90XL robot through 4 poses in each face_object reference frame.
"""

import time
import sys
from pathlib import Path
import numpy as np

# Add simforge to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simforge.config_reader import SimforgeConfig
from simforge.controller.controller import MovementController
from simforge.logging_utils import setup_logging

def main():
    # Setup logging
    logger = setup_logging(debug=False)  # Keep debug disabled for cleaner output

    # Load configuration
    config_path = Path(__file__).parent.parent / "env_configs" / "face_robot.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    logger.info(f"Loading configuration from {config_path}")
    config = SimforgeConfig.from_yaml(str(config_path))

    # Initialize movement controller
    logger.info("Initializing movement controller...")
    controller = MovementController(config, debug=False)

    # Build scene
    logger.info("Building scene...")
    controller.build_scene()

    # Start simulation
    logger.info("Starting simulation...")
    controller.start()

    # Wait for initialization
    time.sleep(2)

    # Define the 4 poses (relative to each face_object frame)
    poses = [
        {"pos": (0.2, 0.2, 0.2), "rpy": (90.0, 0.0, 0.0)},
        {"pos": (-0.2, 0.2, 0.2), "rpy": (90.0, 0.0, 0.0)},
        {"pos": (-0.2, 0.2, -0.2), "rpy": (90.0, 0.0, 0.0)},
        {"pos": (0.2, 0.2, -0.2), "rpy": (90.0, 0.0, 0.0)},
    ]

    # Face object names
    face_objects = [f"face_object_{i}" for i in range(8)]

    logger.info("Starting demo sequence...")

    try:
        for face_obj in face_objects:
            logger.info(f"Moving to poses relative to {face_obj}")

            for i, pose in enumerate(poses):
                logger.info(f"  Pose {i}: position={pose['pos']}, rpy={pose['rpy']}")

                try:
                    # Move robot to pose relative to face_object frame
                    logger.info(f"    Sending move command for pose {i}...")
                    controller.move_cartesian(
                        robot="TX2_90XL_1",
                        position=pose["pos"],
                        orientation_deg=pose["rpy"],
                        frame=f"obj:{face_obj}"
                    )
                    logger.info("    Move command sent successfully")

                    # Wait for movement to complete with up to 2 retries on planning failure
                    max_retries = 2
                    attempt = 0
                    while attempt <= max_retries:
                        if attempt > 0:
                            logger.info(f"    Retrying movement (attempt {attempt+1}/{max_retries+1})...")
                        
                        # Send/Resend the command
                        # Note: move_cartesian is idempotent here; reissuing will replan from current state
                        controller.move_cartesian(
                            robot="TX2_90XL_1",
                            position=pose["pos"],
                            orientation_deg=pose["rpy"],
                            frame=f"obj:{face_obj}"
                        )

                        logger.info("    Waiting for movement to complete...")

                        # Phase 1a: Wait for trajectory to be planned (become active)
                        planning_deadline = time.time() + 10.0
                        saw_active = False
                        while time.time() < planning_deadline:
                            runtime = controller._get_runtime("TX2_90XL_1")
                            if runtime.active_traj is not None:
                                if not saw_active:
                                    logger.info("    Trajectory active...")
                                saw_active = True
                                break
                            time.sleep(0.05)

                        # If we never saw an active trajectory, treat as planning failure and retry
                        if not saw_active:
                            logger.warning("    Planning produced no active trajectory; will retry if attempts remain")
                            attempt += 1
                            if attempt > max_retries:
                                logger.error("    Planning failed after max retries; moving to next pose")
                            continue

                        # Phase 1b: Wait for the active trajectory to finish
                        while True:
                            runtime = controller._get_runtime("TX2_90XL_1")
                            if runtime.active_traj is None:
                                logger.info("    Trajectory finished, waiting for pose validation to settle...")
                                break
                            time.sleep(0.05)

                        # Phase 2: Wait for pose validation to settle (pending -> False stable)
                        stable_false_since = None
                        while True:
                            runtime = controller._get_runtime("TX2_90XL_1")
                            if runtime.pending_pose_validation:
                                stable_false_since = None
                            else:
                                if stable_false_since is None:
                                    stable_false_since = time.time()
                                if time.time() - stable_false_since >= 0.25:
                                    logger.info("    Pose validation settled; movement complete!")
                                    break
                            time.sleep(0.05)

                        # Successful completion for this pose; exit retry loop
                        break
                    
                except Exception as e:
                    logger.error(f"    Failed to execute pose {i} for {face_obj}: {e}")
                    logger.info("    Continuing to next pose...")
                    continue
                    
            logger.info(f"Completed sequence for {face_obj}")
            time.sleep(1)  # Brief pause between face objects

        logger.info("Demo sequence completed!")

        # Keep the simulation running
        logger.info("Simulation running... Press Ctrl+C to exit")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error during demo: {e}")
    finally:
        logger.info("Stopping simulation...")
        controller.stop()

if __name__ == "__main__":
    main()