"""
Valid8 Dual Cell Bringup - Real Robot Mode

Launches the complete Valid8 dual UR5e cell with real robots:
- Robot State Publisher with real hardware
- Controllers (ros2_control with UR driver)
- MoveIt2 for motion planning
- Foxglove Bridge for visualization
- Command Gateway for client communication

Usage:
  ros2 launch valid8_dual_cell_bringup real.launch.py

Arguments:
  nakul_robot_ip:=192.168.1.9    IP of nakul robot
  sahadev_robot_ip:=192.168.1.16  IP of sahadev robot
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Robot IPs
    nakul_robot_ip = LaunchConfiguration('nakul_robot_ip')
    sahadev_robot_ip = LaunchConfiguration('sahadev_robot_ip')
    
    # Launch configurations
    launch_rviz = LaunchConfiguration('launch_rviz')
    launch_foxglove = LaunchConfiguration('launch_foxglove')
    launch_gateway = LaunchConfiguration('launch_gateway')
    launch_moveit = LaunchConfiguration('launch_moveit')
    headless_mode = LaunchConfiguration('headless_mode')
    
    declared_arguments = [
        DeclareLaunchArgument(
            'nakul_robot_ip',
            default_value='192.168.1.9',
            description='IP address of nakul UR5e robot',
        ),
        DeclareLaunchArgument(
            'sahadev_robot_ip',
            default_value='192.168.1.16',
            description='IP address of sahadev UR5e robot',
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='false',
            description='Launch RViz for visualization',
        ),
        DeclareLaunchArgument(
            'launch_foxglove',
            default_value='true',
            description='Launch Foxglove Bridge',
        ),
        DeclareLaunchArgument(
            'launch_gateway',
            default_value='true',
            description='Launch Command Gateway WebSocket server',
        ),
        DeclareLaunchArgument(
            'launch_moveit',
            default_value='true',
            description='Launch MoveIt2 move_group',
        ),
        DeclareLaunchArgument(
            'headless_mode',
            default_value='false',
            description='Enable headless mode (no teach pendant interaction needed)',
        ),
    ]
    
    # Start robots with real hardware
    start_robots = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('valid8_dual_cell_control'),
                'launch',
                'start_robots.launch.py',
            ])
        ),
        launch_arguments={
            'nakul_robot_ip': nakul_robot_ip,
            'sahadev_robot_ip': sahadev_robot_ip,
            'use_fake_hardware': 'false',
            'headless_mode': headless_mode,
            'launch_rviz': 'false',
        }.items(),
    )
    
    # MoveIt2 Move Group (delayed to allow robot startup)
    moveit = TimerAction(
        period=5.0,  # Wait 5 seconds for robot startup
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('valid8_dual_cell_moveit_config'),
                        'launch',
                        'move_group.launch.py',
                    ])
                ),
                launch_arguments={
                    'use_sim': 'false',
                    'launch_rviz': launch_rviz,
                    'nakul_robot_ip': nakul_robot_ip,
                    'sahadev_robot_ip': sahadev_robot_ip,
                }.items(),
            ),
        ],
        condition=IfCondition(launch_moveit),
    )
    
    # Foxglove Bridge
    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[
            {'port': 9090},
            {'address': '0.0.0.0'},
            {'capabilities': ['clientPublish', 'parameters', 'parametersSubscribe', 'services', 'connectionGraph']},
            {'send_buffer_limit': 10000000},
        ],
        condition=IfCondition(launch_foxglove),
    )
    
    # Command Gateway (delayed to allow robot startup)
    gateway = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('simforge_gateway'),
                        'launch',
                        'gateway.launch.py',
                    ])
                ),
                launch_arguments={
                    'websocket_port': '8766',
                }.items(),
            ),
        ],
        condition=IfCondition(launch_gateway),
    )
    
    return LaunchDescription(
        declared_arguments + [
            start_robots,
            moveit,
            foxglove_bridge,
            gateway,
        ]
    )
