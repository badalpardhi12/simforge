#!/usr/bin/env python3
"""
Direct Joint State Publishing Test

Run this INSIDE the Docker container to test the ROS 2 pipeline:
    sudo docker exec -it simforge_server_dev bash
    source /opt/ros/humble/setup.bash
    source /ros2_ws/install/setup.bash
    python3 /ros2_ws/src/simforge_server/test_joint_publish.py

This will publish joint states directly to /joint_states and you should
see the robot move in Foxglove.
"""

import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


class JointPublishTest(Node):
    def __init__(self):
        super().__init__('joint_publish_test')
        
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Home position
        self.home = [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]
        
        self.get_logger().info("Joint Publish Test Node Started")
        self.get_logger().info(f"Publishing to /joint_states")
        self.get_logger().info(f"Joint names: {self.joint_names}")
        
    def publish_joints(self, positions):
        """Publish joint positions."""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.name = self.joint_names
        msg.position = positions
        msg.velocity = [0.0] * 6
        msg.effort = [0.0] * 6
        
        self.joint_pub.publish(msg)
        
    def run_test(self):
        """Run a simple movement test."""
        self.get_logger().info("=" * 60)
        self.get_logger().info("Starting joint publish test...")
        self.get_logger().info("Watch Foxglove - robot should move!")
        self.get_logger().info("=" * 60)
        
        # Define some test poses
        poses = [
            ("Home", self.home),
            ("Pose 1 - Base left", [0.5, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]),
            ("Pose 2 - Base right", [-0.5, -math.pi/2, 0.0, -math.pi/2, 0.0, 0.0]),
            ("Pose 3 - Shoulder up", [0.0, -math.pi/3, 0.0, -math.pi/2, 0.0, 0.0]),
            ("Pose 4 - Elbow bent", [0.0, -math.pi/2, math.pi/2, -math.pi/2, 0.0, 0.0]),
            ("Pose 5 - Wrist rotate", [0.0, -math.pi/2, 0.0, -math.pi/2, 0.0, math.pi/2]),
            ("Home", self.home),
        ]
        
        for i, (name, joints) in enumerate(poses):
            self.get_logger().info(f"Moving to: {name}")
            self.get_logger().info(f"  Joints: {[f'{j:.2f}' for j in joints]}")
            
            # Interpolate to the pose
            start_joints = self.home if i == 0 else poses[i-1][1]
            self._interpolate_to(start_joints, joints, duration=1.0)
            
            # Hold for a moment
            time.sleep(0.5)
            
        self.get_logger().info("=" * 60)
        self.get_logger().info("Test complete!")
        self.get_logger().info("=" * 60)
        
    def _interpolate_to(self, start, end, duration=1.0, rate=50.0):
        """Interpolate from start to end joints."""
        steps = int(duration * rate)
        dt = duration / steps
        
        for step in range(steps + 1):
            t = step / steps
            # Smooth interpolation
            t_smooth = (1 - math.cos(t * math.pi)) / 2
            
            current = [s + (e - s) * t_smooth for s, e in zip(start, end)]
            self.publish_joints(current)
            
            # Also spin to process callbacks
            rclpy.spin_once(self, timeout_sec=0)
            
            time.sleep(dt)


def main():
    rclpy.init()
    
    node = JointPublishTest()
    
    try:
        node.run_test()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
