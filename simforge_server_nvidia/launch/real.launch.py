"""
SimForge Server — Real Robot Mode

Launches:
 - Robot State Publisher with real hardware (UR driver)
 - Controllers (ros2_control with UR driver)
 - Foxglove Bridge for visualization (port 9090)
 - cuRobo Command Gateway (port 8766, delayed 8 s for startup)

Environment is selected by the ENV_CONFIG environment variable.
Robot IPs are passed via environment variables:
  - valid8_dual_ur5e: ROBOT_IP_NAKUL_UR5E, ROBOT_IP_SAHADEV_UR5E
  - face_robot_ur20:  ROBOT_IP_UR20

Usage:
  ros2 launch simforge_gateway_nvidia real.launch.py
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_foxglove = LaunchConfiguration('launch_foxglove')
    launch_gateway = LaunchConfiguration('launch_gateway')
    headless_mode = LaunchConfiguration('headless_mode')

    # Determine environment from ENV_CONFIG
    env_config = os.environ.get('ENV_CONFIG', 'valid8_dual_ur5e')

    declared_arguments = [
        DeclareLaunchArgument(
            'launch_foxglove', default_value='true',
            description='Launch Foxglove Bridge',
        ),
        DeclareLaunchArgument(
            'launch_gateway', default_value='true',
            description='Launch cuRobo Command Gateway WebSocket server',
        ),
        DeclareLaunchArgument(
            'headless_mode', default_value='true',
            description='Enable headless mode (no teach pendant interaction)',
        ),
    ]

    actions = []

    if env_config == 'face_robot_ur20':
        # Single UR20
        robot_ip = os.environ.get('ROBOT_IP_UR20', '0.0.0.0')

        declared_arguments.append(
            DeclareLaunchArgument(
                'robot_ip', default_value=robot_ip,
                description='IP address of UR20 robot',
            ),
        )

        start_robots = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('face_robot_ur20_control'),
                    'launch', 'start_robots.launch.py',
                ])
            ),
            launch_arguments={
                'robot_ip': LaunchConfiguration('robot_ip'),
                'use_fake_hardware': 'false',
                'headless_mode': headless_mode,
                'launch_rviz': 'false',
            }.items(),
        )
    else:
        # Default: valid8_dual_ur5e — dual UR5e
        nakul_ip = os.environ.get('ROBOT_IP_NAKUL_UR5E',
                                  os.environ.get('NAKUL_ROBOT_IP', '192.168.1.9'))
        sahadev_ip = os.environ.get('ROBOT_IP_SAHADEV_UR5E',
                                    os.environ.get('SAHADEV_ROBOT_IP', '192.168.1.16'))

        declared_arguments.extend([
            DeclareLaunchArgument(
                'nakul_robot_ip', default_value=nakul_ip,
                description='IP address of nakul UR5e robot',
            ),
            DeclareLaunchArgument(
                'sahadev_robot_ip', default_value=sahadev_ip,
                description='IP address of sahadev UR5e robot',
            ),
        ])

        start_robots = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('valid8_dual_cell_control'),
                    'launch', 'start_robots.launch.py',
                ])
            ),
            launch_arguments={
                'nakul_robot_ip': LaunchConfiguration('nakul_robot_ip'),
                'sahadev_robot_ip': LaunchConfiguration('sahadev_robot_ip'),
                'use_fake_hardware': 'false',
                'headless_mode': headless_mode,
                'launch_rviz': 'false',
            }.items(),
        )

    actions.append(start_robots)

    # ── Foxglove Bridge ──
    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[
            {'port': 9090},
            {'address': '0.0.0.0'},
            {'capabilities': [
                'clientPublish', 'parameters',
                'parametersSubscribe', 'services', 'connectionGraph',
            ]},
            {'send_buffer_limit': 10000000},
        ],
        condition=IfCondition(launch_foxglove),
    )
    actions.append(foxglove_bridge)

    # ── cuRobo Command Gateway (delayed 8 s to allow robot startup) ──
    gateway = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('simforge_gateway_nvidia'),
                        'launch', 'gateway.launch.py',
                    ])
                ),
                launch_arguments={'websocket_port': '8766'}.items(),
            ),
        ],
        condition=IfCondition(launch_gateway),
    )
    actions.append(gateway)

    return LaunchDescription(declared_arguments + actions)
