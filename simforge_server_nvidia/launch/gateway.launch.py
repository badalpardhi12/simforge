"""
SimForge Gateway Launch File

Launches the command gateway WebSocket server with cuRobo GPU
motion planning.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    websocket_port = LaunchConfiguration('websocket_port')
    websocket_host = LaunchConfiguration('websocket_host')
    max_velocity_scaling = LaunchConfiguration('max_velocity_scaling')
    max_acceleration_scaling = LaunchConfiguration('max_acceleration_scaling')
    interpolation_dt = LaunchConfiguration('interpolation_dt')

    config_dir = PathJoinSubstitution([
        FindPackageShare('simforge_gateway_nvidia'), 'config',
    ])

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
        DeclareLaunchArgument(
            'max_velocity_scaling',
            default_value='0.25',
            description='Max velocity scaling factor (0.0-1.0).',
        ),
        DeclareLaunchArgument(
            'max_acceleration_scaling',
            default_value='0.25',
            description='Max acceleration scaling factor (0.0-1.0).',
        ),
        DeclareLaunchArgument(
            'interpolation_dt',
            default_value='0.02',
            description='cuRobo trajectory interpolation timestep (seconds). '
                        '0.02 = 50 Hz output waypoints.',
        ),
    ]

    command_gateway_node = Node(
        package='simforge_gateway_nvidia',
        executable='command_gateway_curobo_node.py',
        name='command_gateway',
        output='screen',
        parameters=[
            {'websocket_port': websocket_port},
            {'websocket_host': websocket_host},
            {'max_clients': 5},
            {'max_velocity_scaling': max_velocity_scaling},
            {'max_acceleration_scaling': max_acceleration_scaling},
            {'interpolation_dt': interpolation_dt},
            {'config_dir': config_dir},
        ],
    )

    return LaunchDescription(
        declared_arguments + [command_gateway_node]
    )
