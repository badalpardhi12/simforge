#!/usr/bin/env python3
"""
Orchestrator Node

High-level agent that coordinates VLA, perception, and motion planning.

This node:
1. Receives task instructions from Command Gateway
2. Captures current camera image
3. Queries VLA for action plan
4. Executes waypoints via Robot Control with collision checking
5. Monitors execution and handles failures
6. Reports results back to client
"""

import asyncio
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose


class TaskState(IntEnum):
    """Task execution state."""
    IDLE = 0
    CAPTURING_IMAGE = 1
    VLA_INFERENCE = 2
    PLANNING = 3
    EXECUTING = 4
    COMPLETED = 5
    FAILED = 6
    ABORTED = 7


@dataclass
class TaskResult:
    """Result of task execution."""
    success: bool
    message: str
    completed_steps: List[str]
    skipped_steps: List[str]
    total_time_sec: float
    vla_inference_time_sec: float
    motion_time_sec: float
    waypoints_executed: int
    replans_performed: int


class OrchestratorNode(Node):
    """
    Orchestrator Node - coordinates the full manipulation pipeline.
    
    Flow:
    1. Receive text command from Mac client (via Command Gateway)
    2. Get current image from camera
    3. Query VLA for action plan
    4. For each waypoint:
       a. Query cuMotion for collision-free trajectory
       b. Execute trajectory
       c. Handle gripper actions
       d. Check success
    5. Report result to Mac client
    """

    def __init__(self):
        super().__init__('orchestrator')
        
        # Declare parameters
        self.declare_parameter('default_robot', 'ur20')
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('max_velocity_scale', 0.5)
        self.declare_parameter('collision_check_enabled', True)
        self.declare_parameter('max_replans', 3)
        
        self.default_robot = self.get_parameter('default_robot').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.max_velocity_scale = self.get_parameter('max_velocity_scale').value
        self.collision_check = self.get_parameter('collision_check_enabled').value
        self.max_replans = self.get_parameter('max_replans').value
        
        # Callback group for async operations
        self.callback_group = ReentrantCallbackGroup()
        
        # Task state
        self.current_task_state = TaskState.IDLE
        self.current_task_instruction = ""
        self.current_task_start_time = 0.0
        
        # Latest camera image
        self.latest_image: Optional[Image] = None
        self.latest_image_time = None
        
        # === Subscribers ===
        # Camera image
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        
        # VLA status
        self.vla_status_sub = self.create_subscription(
            String,
            '/vla/status',
            self.vla_status_callback,
            10
        )
        
        # Robot state
        self.robot_state_sub = self.create_subscription(
            String,
            f'/{self.default_robot}/robot_state',
            self.robot_state_callback,
            10
        )
        
        # === Publishers ===
        # Orchestrator status
        self.status_pub = self.create_publisher(
            String,
            '/orchestrator/status',
            10
        )
        
        # Task progress
        self.progress_pub = self.create_publisher(
            String,
            '/orchestrator/task_progress',
            10
        )
        
        # === Service Clients ===
        # We'll simulate VLA and motion services for now
        # In production, these would be proper service/action clients
        
        # === Timers ===
        # Status publishing
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info(
            f"Orchestrator Node initialized - robot: {self.default_robot}"
        )

    def image_callback(self, msg: Image):
        """Store latest camera image."""
        self.latest_image = msg
        self.latest_image_time = self.get_clock().now()

    def vla_status_callback(self, msg: String):
        """Process VLA status updates."""
        # Parse and log VLA status
        pass

    def robot_state_callback(self, msg: String):
        """Process robot state updates."""
        # Parse and check for errors
        if 'pstop:True' in msg.data or 'estop:True' in msg.data:
            if self.current_task_state in [TaskState.EXECUTING, TaskState.PLANNING]:
                self.get_logger().error("Robot stopped during task execution!")
                self.current_task_state = TaskState.FAILED

    async def execute_task(
        self,
        instruction: str,
        robot_name: Optional[str] = None,
        max_velocity_scale: Optional[float] = None,
    ) -> TaskResult:
        """
        Execute a VLA-guided manipulation task.
        
        Args:
            instruction: Natural language instruction
            robot_name: Robot to use (default from config)
            max_velocity_scale: Maximum velocity (default from config)
            
        Returns:
            TaskResult with execution details
        """
        robot = robot_name or self.default_robot
        velocity_scale = max_velocity_scale or self.max_velocity_scale
        
        self.current_task_instruction = instruction
        self.current_task_start_time = time.time()
        
        completed_steps = []
        skipped_steps = []
        vla_time = 0.0
        motion_time = 0.0
        waypoints_executed = 0
        replans = 0
        
        try:
            # Step 1: Capture image
            self.current_task_state = TaskState.CAPTURING_IMAGE
            self.publish_progress("Capturing image...", 0.05)
            
            if self.latest_image is None:
                # Wait for image (up to 2 seconds)
                for _ in range(20):
                    if self.latest_image is not None:
                        break
                    await asyncio.sleep(0.1)
                
                if self.latest_image is None:
                    raise RuntimeError("No camera image available")
            
            # Step 2: VLA inference
            self.current_task_state = TaskState.VLA_INFERENCE
            self.publish_progress("Running VLA inference...", 0.1)
            
            vla_start = time.time()
            waypoints, labels, gripper_actions = await self.get_vla_prediction(
                self.latest_image,
                instruction
            )
            vla_time = time.time() - vla_start
            
            self.get_logger().info(
                f"VLA predicted {len(waypoints)} waypoints: {labels}"
            )
            
            # Step 3: Execute waypoints
            self.current_task_state = TaskState.EXECUTING
            total_waypoints = len(waypoints)
            
            for i, (waypoint, label, gripper) in enumerate(
                zip(waypoints, labels, gripper_actions)
            ):
                step_progress = 0.2 + (0.7 * i / total_waypoints)
                self.publish_progress(f"Executing: {label}", step_progress)
                
                # Step 3a: Plan motion
                self.current_task_state = TaskState.PLANNING
                motion_start = time.time()
                
                # Execute motion (simulated for now)
                success = await self.execute_motion(
                    robot,
                    waypoint,
                    velocity_scale,
                    self.collision_check
                )
                
                if not success:
                    # Try replanning
                    if replans < self.max_replans:
                        replans += 1
                        self.get_logger().warn(f"Replanning (attempt {replans})")
                        success = await self.execute_motion(
                            robot, waypoint, velocity_scale * 0.5, True
                        )
                    
                    if not success:
                        self.get_logger().error(f"Motion failed at step: {label}")
                        skipped_steps.extend(labels[i:])
                        raise RuntimeError(f"Motion failed at step: {label}")
                
                motion_time += time.time() - motion_start
                
                # Step 3b: Gripper action
                if abs(gripper) > 0.5:
                    action = "close" if gripper > 0 else "open"
                    self.get_logger().info(f"Gripper {action}")
                    await self.execute_gripper_action(gripper)
                
                completed_steps.append(label)
                waypoints_executed += 1
                
                self.current_task_state = TaskState.EXECUTING
            
            # Success!
            self.current_task_state = TaskState.COMPLETED
            self.publish_progress("Task completed", 1.0)
            
            return TaskResult(
                success=True,
                message=f"Task completed: {instruction}",
                completed_steps=completed_steps,
                skipped_steps=skipped_steps,
                total_time_sec=time.time() - self.current_task_start_time,
                vla_inference_time_sec=vla_time,
                motion_time_sec=motion_time,
                waypoints_executed=waypoints_executed,
                replans_performed=replans,
            )
            
        except Exception as e:
            self.current_task_state = TaskState.FAILED
            self.get_logger().error(f"Task failed: {e}")
            
            return TaskResult(
                success=False,
                message=str(e),
                completed_steps=completed_steps,
                skipped_steps=skipped_steps,
                total_time_sec=time.time() - self.current_task_start_time,
                vla_inference_time_sec=vla_time,
                motion_time_sec=motion_time,
                waypoints_executed=waypoints_executed,
                replans_performed=replans,
            )

    async def get_vla_prediction(
        self,
        image: Image,
        instruction: str
    ) -> tuple:
        """
        Get VLA action prediction.
        
        In production, this would call the VLA inference service.
        For now, simulate the response.
        """
        # Simulate VLA inference time
        await asyncio.sleep(0.1)
        
        # Generate simulated waypoints based on instruction
        instruction_lower = instruction.lower()
        
        if "pick" in instruction_lower:
            waypoints = [
                self._make_pose(0.4, 0.0, 0.3),
                self._make_pose(0.4, 0.0, 0.15),
                self._make_pose(0.4, 0.0, 0.35),
            ]
            labels = ["approach", "grasp", "lift"]
            grippers = [-1.0, 1.0, 0.0]
        elif "place" in instruction_lower:
            waypoints = [
                self._make_pose(0.4, 0.2, 0.3),
                self._make_pose(0.4, 0.2, 0.15),
                self._make_pose(0.4, 0.2, 0.25),
            ]
            labels = ["approach", "lower", "release"]
            grippers = [0.0, 0.0, -1.0]
        else:
            waypoints = [self._make_pose(0.35, 0.0, 0.25)]
            labels = ["move"]
            grippers = [0.0]
        
        return waypoints, labels, grippers

    async def execute_motion(
        self,
        robot_name: str,
        target_pose: Pose,
        velocity_scale: float,
        collision_check: bool
    ) -> bool:
        """
        Execute motion to target pose.
        
        In production, this would call the MoveRobot action server.
        """
        self.get_logger().info(
            f"Moving to ({target_pose.position.x:.2f}, "
            f"{target_pose.position.y:.2f}, {target_pose.position.z:.2f})"
        )
        
        # Simulate motion time
        await asyncio.sleep(0.5)
        
        # 95% success rate in simulation
        import random
        return random.random() < 0.95

    async def execute_gripper_action(self, gripper_command: float) -> bool:
        """
        Execute gripper action.
        
        Args:
            gripper_command: -1.0 = open, 1.0 = close
        """
        action = "close" if gripper_command > 0 else "open"
        self.get_logger().info(f"Gripper {action}")
        
        # Simulate gripper action time
        await asyncio.sleep(0.2)
        return True

    def _make_pose(self, x: float, y: float, z: float) -> Pose:
        """Create a pose with default orientation."""
        from geometry_msgs.msg import Point, Quaternion
        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pose.orientation = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)
        return pose

    def publish_progress(self, status: str, progress: float):
        """Publish task progress."""
        msg = String()
        msg.data = (
            f"instruction:{self.current_task_instruction}|"
            f"status:{status}|"
            f"progress:{progress:.2f}|"
            f"state:{self.current_task_state.name}"
        )
        self.progress_pub.publish(msg)

    def publish_status(self):
        """Publish orchestrator status."""
        msg = String()
        msg.data = (
            f"state:{self.current_task_state.name}|"
            f"robot:{self.default_robot}|"
            f"has_image:{self.latest_image is not None}"
        )
        self.status_pub.publish(msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = OrchestratorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Orchestrator Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
