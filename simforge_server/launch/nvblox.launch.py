#!/usr/bin/env python3
"""
nvblox Launch Configuration for Simforge

Launches nvblox for real-time 3D reconstruction and ESDF generation.
Used by cuMotion for GPU-accelerated collision avoidance.

This launch file supports:
- Single depth camera (RealSense D455)
- Multiple depth cameras (future)
- ESDF generation for collision checking
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for nvblox."""
    
    # Launch arguments
    use_sim = LaunchConfiguration('use_sim', default='false')
    camera_frame = LaunchConfiguration('camera_frame', default='camera_link')
    global_frame = LaunchConfiguration('global_frame', default='world')
    
    # Declare launch arguments
    declare_use_sim = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Use simulated depth camera'
    )
    
    declare_camera_frame = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_link',
        description='Camera optical frame'
    )
    
    declare_global_frame = DeclareLaunchArgument(
        'global_frame',
        default_value='world',
        description='Global/world frame for ESDF'
    )
    
    # nvblox mapper node
    nvblox_node = Node(
        package='nvblox_ros',
        executable='nvblox_node',
        name='nvblox_node',
        output='screen',
        parameters=[{
            # Frame configuration
            'global_frame': global_frame,
            'voxel_size': 0.02,  # 2cm voxels for good resolution
            
            # ESDF configuration
            'compute_esdf': True,
            'esdf_distance_slice': True,  # Publish 2D slice for visualization
            'esdf_2d_min_height': 0.0,
            'esdf_2d_max_height': 1.5,
            
            # Mesh configuration (for visualization)
            'compute_mesh': True,
            'mesh_update_rate_hz': 5.0,
            
            # Integration parameters
            'max_tsdf_distance': 0.1,  # Truncation distance
            'max_integration_distance_m': 5.0,  # Max depth integration
            
            # Performance tuning
            'tsdf_integrator_max_integration_distance_m': 5.0,
            'color_integrator_max_integration_distance_m': 5.0,
            
            # Decay configuration (for dynamic environments)
            'use_tsdf_decay': True,
            'tsdf_decay_rate': 0.1,
            
            # CUDA memory settings
            'esdf_integrator_min_weight': 0.0001,
            
            # Layer update rates
            'esdf_update_rate_hz': 10.0,
            'occupancy_publication_rate_hz': 2.0,
        }],
        remappings=[
            # Input depth image
            ('depth/image', '/camera/depth/image_rect_raw'),
            ('depth/camera_info', '/camera/depth/camera_info'),
            # Optional color for mesh visualization
            ('color/image', '/camera/color/image_raw'),
            ('color/camera_info', '/camera/color/camera_info'),
            # Output ESDF for cuMotion
            ('esdf', '/nvblox/esdf'),
            ('combined_esdf', '/nvblox/combined_esdf'),
            # Mesh for visualization
            ('mesh', '/nvblox/mesh'),
            ('mesh_marker', '/nvblox/mesh_marker'),
        ],
    )
    
    # Static transform from camera to robot base (example)
    # In production, this comes from robot calibration
    static_camera_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_base_link',
        arguments=[
            # x y z qx qy qz qw frame_id child_frame_id
            '0.5', '0.0', '1.0',  # Position: 50cm forward, 1m up from base
            '0.0', '0.707', '0.0', '0.707',  # Quaternion: looking down at 90 degrees
            'base_link', 'camera_link'
        ],
    )
    
    return LaunchDescription([
        declare_use_sim,
        declare_camera_frame,
        declare_global_frame,
        nvblox_node,
        static_camera_transform,
    ])
