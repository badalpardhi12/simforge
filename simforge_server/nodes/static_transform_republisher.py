#!/usr/bin/env python3
"""
Static Transform Republisher Node

This node periodically republishes static transforms to /tf to ensure
visualization tools like Foxglove Studio receive them reliably.

The issue: /tf_static uses transient local QoS (latched), but websocket
bridges may not properly forward this. By periodically publishing to /tf,
we ensure all frames are always available.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener
import math


class StaticTransformRepublisher(Node):
    """Republishes static transforms periodically for reliable visualization."""

    def __init__(self):
        super().__init__('static_tf_republisher')
        
        # Parameters
        self.declare_parameter('publish_rate', 5.0)  # Hz - higher rate for responsiveness
        
        publish_rate = self.get_parameter('publish_rate').value
        
        # TF Buffer and Listener to capture static transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publisher for /tf (dynamic transforms topic)
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)
        
        # Subscriber for /tf_static to capture static transforms
        static_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.static_transforms = []
        self.tf_static_sub = self.create_subscription(
            TFMessage,
            '/tf_static',
            self.tf_static_callback,
            static_qos
        )
        
        # Timer for periodic republishing
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_transforms)
        
        self.get_logger().info(
            f'Static Transform Republisher initialized - rate={publish_rate}Hz'
        )

    def tf_static_callback(self, msg: TFMessage):
        """Capture static transforms as they're published."""
        for transform in msg.transforms:
            # Check if we already have this transform
            found = False
            for i, existing in enumerate(self.static_transforms):
                if (existing.header.frame_id == transform.header.frame_id and
                    existing.child_frame_id == transform.child_frame_id):
                    # Update existing
                    self.static_transforms[i] = transform
                    found = True
                    break
            if not found:
                self.static_transforms.append(transform)
                self.get_logger().info(
                    f'Captured static TF: {transform.header.frame_id} -> {transform.child_frame_id}'
                )

    def publish_transforms(self):
        """Periodically publish static transforms to /tf."""
        if not self.static_transforms:
            return
            
        # Update timestamps and publish
        msg = TFMessage()
        current_time = self.get_clock().now().to_msg()
        
        for transform in self.static_transforms:
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = transform.header.frame_id
            t.child_frame_id = transform.child_frame_id
            t.transform = transform.transform
            msg.transforms.append(t)
        
        self.tf_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StaticTransformRepublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


def main(args=None):
    rclpy.init(args=args)
    node = StaticTransformRepublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
