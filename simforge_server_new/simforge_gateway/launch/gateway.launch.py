"""
SimForge Gateway Launch File

Launches the command gateway WebSocket server and robot state publisher.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    websocket_port = LaunchConfiguration('websocket_port')
    websocket_host = LaunchConfiguration('websocket_host')
    
    declared_arguments = [
        DeclareLaunchArgument(
            'websocket_port',
            default_value='8766',
            description='WebSocket server port',
        ),
        DeclareLaunchArgument(
            'websocket_host',
            default_value='0.0.0.0',
            description='WebSocket server host',
        ),
    ]
    
    command_gateway_node = Node(
        package='simforge_gateway',
        executable='command_gateway_node.py',
        name='command_gateway',
        output='screen',
        parameters=[
            {'websocket_port': websocket_port},
            {'websocket_host': websocket_host},
            {'max_clients': 5},
        ],
    )
    
    return LaunchDescription(
        declared_arguments + [
            command_gateway_node,
        ]
    )
