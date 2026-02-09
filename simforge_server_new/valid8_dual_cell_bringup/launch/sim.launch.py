"""
Valid8 Dual Cell Bringup - Simulation Mode

Launches the complete Valid8 dual UR5e cell in simulation mode with:
- Robot State Publisher with mock hardware
- Controllers (ros2_control)
- MoveIt2 for motion planning
- Foxglove Bridge for visualization
- Command Gateway for client communication

Usage:
  ros2 launch valid8_dual_cell_bringup sim.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch configurations
    launch_rviz = LaunchConfiguration('launch_rviz')
    launch_foxglove = LaunchConfiguration('launch_foxglove')
    launch_gateway = LaunchConfiguration('launch_gateway')
    launch_moveit = LaunchConfiguration('launch_moveit')
    
    declared_arguments = [
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
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
    ]
    
    # Set use_sim_time for all nodes
    use_sim_time = SetParameter(name='use_sim_time', value=False)
    
    # Start robots with fake hardware (simulation)
    start_robots = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('valid8_dual_cell_control'),
                'launch',
                'start_robots.launch.py',
            ])
        ),
        launch_arguments={
            'use_fake_hardware': 'true',
            'launch_rviz': 'false',  # We'll launch our own RViz
        }.items(),
    )
    
    # MoveIt2 Move Group
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('valid8_dual_cell_moveit_config'),
                'launch',
                'move_group.launch.py',
            ])
        ),
        launch_arguments={
            'use_sim': 'true',
            'launch_rviz': launch_rviz,
        }.items(),
        condition=IfCondition(launch_moveit),
    )
    
    # Foxglove Bridge - serves ALL ROS2 topics over WebSocket on port 9090
    # Connect from Foxglove Studio on macOS: ws://<server-ip>:9090
    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[
            {'port': 9090},
            {'address': '0.0.0.0'},
            {'send_buffer_limit': 100000000},  # 100MB buffer for mesh/pointcloud data
            {'use_compression': True},
            {'asset_uri_allowlist': ['package://.*']},  # Allow URDF mesh loading
        ],
        condition=IfCondition(launch_foxglove),
    )
    
    # Command Gateway
    gateway = IncludeLaunchDescription(
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
        condition=IfCondition(launch_gateway),
    )
    
    return LaunchDescription(
        declared_arguments + [
            use_sim_time,
            start_robots,
            moveit,
            foxglove_bridge,
            gateway,
        ]
    )
