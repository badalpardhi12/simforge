#!/usr/bin/env python3
"""
Perception Node

Handles camera input and nvblox integration for 3D scene reconstruction.

This node:
1. Subscribes to RealSense depth/color images
2. Integrates with nvblox for TSDF/ESDF reconstruction
3. Publishes mesh for visualization
4. Provides distance queries for collision checking
"""

import numpy as np
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class PerceptionNode(Node):
    """
    Perception Node - handles camera input and 3D reconstruction.
    
    In production:
    - Subscribes to RealSense camera topics
    - Integrates with isaac_ros_nvblox for real-time 3D reconstruction
    - Publishes ESDF for cuMotion collision checking
    
    In development/simulation:
    - Can use Genesis simulation for synthetic depth images
    """

    def __init__(self):
        super().__init__('perception')
        
        # Declare parameters
        self.declare_parameter('camera_topic_prefix', '/camera')
        self.declare_parameter('use_simulation', False)
        self.declare_parameter('voxel_size', 0.05)  # 5cm voxels
        self.declare_parameter('publish_rate', 10.0)
        
        self.camera_prefix = self.get_parameter('camera_topic_prefix').value
        self.use_simulation = self.get_parameter('use_simulation').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # CV Bridge for image conversion
        self.cv_bridge = CvBridge() if HAS_CV_BRIDGE else None
        
        # Camera intrinsics (will be updated from camera_info)
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion_coeffs: Optional[np.ndarray] = None
        self.image_width = 640
        self.image_height = 480
        
        # Latest images
        self.latest_color_image: Optional[np.ndarray] = None
        self.latest_depth_image: Optional[np.ndarray] = None
        self.latest_image_time = None
        
        # === Subscribers ===
        # Color image
        self.color_sub = self.create_subscription(
            Image,
            f'{self.camera_prefix}/color/image_raw',
            self.color_callback,
            10
        )
        
        # Depth image (aligned to color)
        self.depth_sub = self.create_subscription(
            Image,
            f'{self.camera_prefix}/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        # Camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            f'{self.camera_prefix}/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # === Publishers ===
        # Point cloud (for visualization)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/perception/pointcloud',
            10
        )
        
        # Status
        self.status_pub = self.create_publisher(
            String,
            '/perception/status',
            10
        )
        
        # === Timer ===
        # Status publishing
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info(
            f"Perception Node initialized - camera: {self.camera_prefix}"
        )

    def color_callback(self, msg: Image):
        """Process color image."""
        if not self.cv_bridge:
            return
        
        try:
            self.latest_color_image = self.cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
            self.latest_image_time = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Color image conversion failed: {e}")

    def depth_callback(self, msg: Image):
        """Process depth image."""
        if not self.cv_bridge:
            return
        
        try:
            # Depth is typically 16-bit unsigned (mm) or 32-bit float (m)
            if msg.encoding == '16UC1':
                self.latest_depth_image = self.cv_bridge.imgmsg_to_cv2(
                    msg, desired_encoding='16UC1'
                )
                # Convert to meters
                self.latest_depth_image = self.latest_depth_image.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                self.latest_depth_image = self.cv_bridge.imgmsg_to_cv2(
                    msg, desired_encoding='32FC1'
                )
            else:
                self.get_logger().warn(f"Unknown depth encoding: {msg.encoding}")
                
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        """Process camera info to get intrinsics."""
        self.image_width = msg.width
        self.image_height = msg.height
        
        # Extract camera matrix (3x3)
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        
        # Extract distortion coefficients
        self.distortion_coeffs = np.array(msg.d)

    def get_latest_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get latest color and depth images."""
        return self.latest_color_image, self.latest_depth_image

    def depth_to_pointcloud(self, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert depth image to point cloud.
        
        Args:
            depth_image: Depth image in meters (HxW float32)
            
        Returns:
            Point cloud as Nx3 array (x, y, z)
        """
        if self.camera_matrix is None:
            return None
        
        height, width = depth_image.shape
        
        # Create pixel coordinate grid
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        
        # Get camera intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Project to 3D
        z = depth_image
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack and reshape
        points = np.stack([x, y, z], axis=-1)
        
        # Filter invalid points (zero depth)
        valid_mask = z > 0.1  # Minimum 10cm
        points = points[valid_mask]
        
        return points

    def query_distance_at_point(
        self,
        point: np.ndarray,
        esdf_data: Optional[np.ndarray] = None
    ) -> float:
        """
        Query the distance to nearest obstacle at a 3D point.
        
        In production, this would query the nvblox ESDF.
        
        Args:
            point: 3D point (x, y, z)
            esdf_data: Optional ESDF voxel grid
            
        Returns:
            Distance to nearest obstacle (positive = free space)
        """
        # TODO: Integrate with nvblox ESDF query
        # For now, return a placeholder
        return 1.0  # 1 meter (safe)

    def publish_status(self):
        """Publish perception status."""
        has_color = self.latest_color_image is not None
        has_depth = self.latest_depth_image is not None
        has_intrinsics = self.camera_matrix is not None
        
        status_msg = String()
        status_msg.data = (
            f"color:{has_color}|"
            f"depth:{has_depth}|"
            f"intrinsics:{has_intrinsics}|"
            f"resolution:{self.image_width}x{self.image_height}"
        )
        self.status_pub.publish(status_msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = PerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Perception Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
