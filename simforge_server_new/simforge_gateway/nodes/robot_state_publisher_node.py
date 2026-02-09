#!/usr/bin/env python3
"""
Robot State Publisher Node

Publishes robot state information for the valid8 dual cell setup.
Aggregates joint states and TF data for both robots.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import String
import json
import time


class RobotStatePublisherNode(Node):
    """Node that aggregates and publishes robot state information."""

    def __init__(self):
        super().__init__('robot_state_aggregator')
        
        # Robot configurations
        self.robots = {
            'nakul': {
                'prefix': 'nakul_',
                'joints': [],
                'positions': [],
                'velocities': [],
                'last_update': 0.0,
            },
            'sahadev': {
                'prefix': 'sahadev_',
                'joints': [],
                'positions': [],
                'velocities': [],
                'last_update': 0.0,
            },
        }
        
        # Subscribe to joint states
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos
        )
        
        # Publisher for aggregated state
        self.state_pub = self.create_publisher(
            String,
            '/valid8/robot_states',
            10
        )
        
        # Timer for state publishing
        self.create_timer(0.1, self.publish_state)  # 10 Hz
        
        self.get_logger().info('Robot State Aggregator initialized')

    def joint_state_callback(self, msg: JointState):
        """Process incoming joint states."""
        for robot_name, robot_info in self.robots.items():
            prefix = robot_info['prefix']
            
            joints = []
            positions = []
            velocities = []
            
            for i, joint_name in enumerate(msg.name):
                if joint_name.startswith(prefix):
                    joints.append(joint_name)
                    positions.append(msg.position[i] if i < len(msg.position) else 0.0)
                    velocities.append(msg.velocity[i] if i < len(msg.velocity) else 0.0)
            
            if joints:
                robot_info['joints'] = joints
                robot_info['positions'] = positions
                robot_info['velocities'] = velocities
                robot_info['last_update'] = time.time()

    def publish_state(self):
        """Publish aggregated robot state."""
        state = {
            'timestamp': time.time(),
            'robots': {},
        }
        
        for robot_name, robot_info in self.robots.items():
            state['robots'][robot_name] = {
                'joints': robot_info['joints'],
                'positions': robot_info['positions'],
                'velocities': robot_info['velocities'],
                'connected': time.time() - robot_info['last_update'] < 1.0,
            }
        
        msg = String()
        msg.data = json.dumps(state)
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
