"""
Perception Launch File

Launches perception components:
- RealSense camera driver
- Perception node
- (Optional) nvblox for 3D reconstruction

Usage:
    ros2 launch simforge_server perception.launch.py
    ros2 launch simforge_server perception.launch.py use_nvblox:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for perception."""
    
    # Arguments
    use_nvblox_arg = DeclareLaunchArgument(
        'use_nvblox',
        default_value='false',
        description='Enable nvblox for 3D reconstruction'
    )
    
    camera_serial_arg = DeclareLaunchArgument(
        'camera_serial',
        default_value='',
        description='RealSense camera serial number (empty for first available)'
    )
    
    simulation_mode_arg = DeclareLaunchArgument(
        'simulation_mode',
        default_value='false',
        description='Use simulation instead of real camera'
    )
    
    # RealSense camera node
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense_camera',
        output='screen',
        parameters=[{
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'align_depth.enable': True,
            'pointcloud.enable': False,
            'initial_reset': True,
        }],
        remappings=[
            ('/camera/camera/depth/image_rect_raw', '/camera/depth/image'),
            ('/camera/camera/color/image_raw', '/camera/color/image'),
            ('/camera/camera/color/camera_info', '/camera/color/camera_info'),
        ],
    )
    
    # Perception node
    perception_node = Node(
        package='simforge_server',
        executable='perception_node.py',
        name='perception',
        output='screen',
        parameters=[{
            'camera_topic_prefix': '/camera',
            'use_simulation': LaunchConfiguration('simulation_mode'),
            'voxel_size': 0.05,
        }],
    )
    
    # nvblox node (optional)
    nvblox_node = Node(
        package='isaac_ros_nvblox',
        executable='nvblox_node',
        name='nvblox',
        output='screen',
        parameters=[{
            'voxel_size': 0.05,
            'esdf': True,
            'esdf_2d': False,
            'esdf_update_rate_hz': 10.0,
            'mesh': True,
            'mesh_update_rate_hz': 5.0,
            'global_frame': 'world',
            'use_depth_map': True,
        }],
        remappings=[
            ('depth/image', '/camera/depth/image'),
            ('depth/camera_info', '/camera/depth/camera_info'),
            ('color/image', '/camera/color/image'),
            ('color/camera_info', '/camera/color/camera_info'),
        ],
        condition=IfCondition(LaunchConfiguration('use_nvblox')),
    )
    
    return LaunchDescription([
        use_nvblox_arg,
        camera_serial_arg,
        simulation_mode_arg,
        realsense_node,
        perception_node,
        nvblox_node,
    ])
