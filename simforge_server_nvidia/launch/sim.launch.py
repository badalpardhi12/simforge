"""
Valid8 Dual Cell Bringup — Simulation Mode (NVIDIA cuRobo backend)

Launches:
 - Robot State Publisher with mock hardware
 - Controllers (ros2_control — same as MoveIt version)
 - Foxglove Bridge for visualization (port 9090)
 - cuRobo Command Gateway (port 8766)

NOT launched:
 - MoveIt2 move_group (replaced by cuRobo inside gateway)

Usage:
  ros2 launch simforge_gateway_nvidia sim.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_foxglove = LaunchConfiguration('launch_foxglove')
    launch_gateway = LaunchConfiguration('launch_gateway')

    declared_arguments = [
        DeclareLaunchArgument(
            'launch_foxglove', default_value='true',
            description='Launch Foxglove Bridge',
        ),
        DeclareLaunchArgument(
            'launch_gateway', default_value='true',
            description='Launch cuRobo Command Gateway WebSocket server',
        ),
    ]

    use_sim_time = SetParameter(name='use_sim_time', value=False)

    # ── Start robots with fake hardware (same as MoveIt version) ──
    start_robots = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('valid8_dual_cell_control'),
                'launch', 'start_robots.launch.py',
            ])
        ),
        launch_arguments={
            'use_fake_hardware': 'true',
            'launch_rviz': 'false',
        }.items(),
    )

    # ── Foxglove Bridge — serves ALL ROS2 topics on port 9090 ──
    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[
            {'port': 9090},
            {'address': '0.0.0.0'},
            {'send_buffer_limit': 100000000},
            {'use_compression': True},
            {'asset_uri_allowlist': ['package://.*']},
        ],
        condition=IfCondition(launch_foxglove),
    )

    # ── cuRobo Command Gateway ──
    gateway = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('simforge_gateway_nvidia'),
                'launch', 'gateway.launch.py',
            ])
        ),
        launch_arguments={'websocket_port': '8766'}.items(),
        condition=IfCondition(launch_gateway),
    )

    return LaunchDescription(
        declared_arguments + [
            use_sim_time,
            start_robots,
            foxglove_bridge,
            gateway,
        ]
    )
