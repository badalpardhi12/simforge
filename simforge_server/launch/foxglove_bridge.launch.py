"""
Foxglove Bridge Launch File

Launches only the Foxglove Bridge for ROS 2 visualization.

Usage:
    ros2 launch simforge_server foxglove_bridge.launch.py
    ros2 launch simforge_server foxglove_bridge.launch.py port:=9090
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Foxglove Bridge."""
    
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='9090',
        description='WebSocket port for Foxglove Bridge'
    )
    
    address_arg = DeclareLaunchArgument(
        'address',
        default_value='0.0.0.0',
        description='Address to bind to (0.0.0.0 for all interfaces)'
    )
    
    send_buffer_limit_arg = DeclareLaunchArgument(
        'send_buffer_limit',
        default_value='100000000',
        description='Send buffer limit in bytes (100MB default for point clouds)'
    )
    
    use_compression_arg = DeclareLaunchArgument(
        'use_compression',
        default_value='true',
        description='Enable compression for WebSocket messages'
    )
    
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('port'),
            'address': LaunchConfiguration('address'),
            'send_buffer_limit': LaunchConfiguration('send_buffer_limit'),
            'use_compression': LaunchConfiguration('use_compression'),
        }],
    )
    
    return LaunchDescription([
        port_arg,
        address_arg,
        send_buffer_limit_arg,
        use_compression_arg,
        foxglove_bridge_node,
    ])
