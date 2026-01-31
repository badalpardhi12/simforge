"""
Robot Bringup Launch File

Launches only robot control components (without VLA/Perception):
- Safety Watchdog
- Robot Control
- Command Gateway
- Foxglove Bridge

Usage:
    ros2 launch simforge_server robot_bringup.launch.py
    ros2 launch simforge_server robot_bringup.launch.py robot_ip:=192.168.1.10
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for robot bringup."""
    
    # Declare launch arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.10',
        description='IP address of the UR robot'
    )
    
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='ur20',
        description='Name of the robot'
    )
    
    simulation_mode_arg = DeclareLaunchArgument(
        'simulation_mode',
        default_value='false',
        description='Run in simulation mode'
    )
    
    # Get launch configurations
    robot_ip = LaunchConfiguration('robot_ip')
    robot_name = LaunchConfiguration('robot_name')
    simulation_mode = LaunchConfiguration('simulation_mode')
    
    # === Nodes ===
    
    safety_watchdog_node = Node(
        package='simforge_server',
        executable='safety_watchdog_node.py',
        name='safety_watchdog',
        output='screen',
        parameters=[{
            'heartbeat_timeout_sec': 0.1,
            'max_consecutive_misses': 3,
        }],
    )
    
    robot_control_node = Node(
        package='simforge_server',
        executable='robot_control_node.py',
        name='robot_control',
        output='screen',
        parameters=[{
            'robot_ip': robot_ip,
            'robot_name': robot_name,
            'simulation_mode': simulation_mode,
        }],
    )
    
    command_gateway_node = Node(
        package='simforge_server',
        executable='command_gateway_node.py',
        name='command_gateway',
        output='screen',
        parameters=[{
            'websocket_port': 8765,
            'websocket_host': '0.0.0.0',
        }],
    )
    
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': 9090,
            'address': '0.0.0.0',
        }],
    )
    
    return LaunchDescription([
        robot_ip_arg,
        robot_name_arg,
        simulation_mode_arg,
        safety_watchdog_node,
        robot_control_node,
        command_gateway_node,
        foxglove_bridge_node,
    ])
