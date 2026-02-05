"""
Simulation Stack Launch File

A simplified launch file that works without UR Robot Driver package.
For simulation/visualization only.

Launches:
- Safety Watchdog
- Robot State Publisher (for TF and URDF visualization)
- Command Gateway
- MoveIt move_group
- Foxglove Bridge

Usage:
    ros2 launch simforge_server sim_stack.launch.py
"""

import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for simulation stack."""
    
    # Get package share directory for asset paths
    pkg_share = get_package_share_directory('simforge_server')
    
    # Declare launch arguments
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='nakul_ur5e',
        description='Name of the robot'
    )
    
    websocket_port_arg = DeclareLaunchArgument(
        'websocket_port',
        default_value='8766',
        description='WebSocket port for Command Gateway'
    )
    
    foxglove_port_arg = DeclareLaunchArgument(
        'foxglove_port',
        default_value='9090',
        description='WebSocket port for Foxglove Bridge'
    )
    
    # Default URDF path - combined environment URDF with robot, table, and face
    default_urdf = os.path.join(pkg_share, 'assets', 'valid8_environment.urdf')
    
    urdf_path_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value=default_urdf,
        description='Path to combined environment URDF file (robot + table + face)'
    )
    
    # Get launch configurations
    robot_name = LaunchConfiguration('robot_name')
    websocket_port = LaunchConfiguration('websocket_port')
    foxglove_port = LaunchConfiguration('foxglove_port')
    urdf_path = LaunchConfiguration('urdf_path')
    
    # === Nodes ===
    
    # 1. Safety Watchdog
    safety_watchdog_node = Node(
        package='simforge_server',
        executable='safety_watchdog_node.py',
        name='safety_watchdog',
        output='screen',
        parameters=[{
            'heartbeat_timeout_sec': 2.0,
            'max_consecutive_misses': 5,
            'check_frequency_hz': 10.0,
            'enable_force_monitoring': False,
        }],
    )
    
    # 2. Robot State Publisher (publishes TF transforms from joint_states + URDF)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['cat ', urdf_path]),
                value_type=str
            ),
            'publish_frequency': 50.0,
        }],
    )
    
    # 3. Static Transform Republisher (republishes /tf_static to /tf for Foxglove)
    static_tf_republisher = Node(
        package='simforge_server',
        executable='static_transform_republisher.py',
        name='static_tf_republisher',
        output='screen',
        parameters=[{
            'publish_rate': 5.0,
        }],
    )
    
    # 4. Command Gateway
    command_gateway_node = Node(
        package='simforge_server',
        executable='command_gateway_node.py',
        name='command_gateway',
        output='screen',
        parameters=[{
            'websocket_port': websocket_port,
            'websocket_host': '0.0.0.0',
            'max_clients': 5,
            'robot_name': robot_name,
        }],
    )
    
    # 5. MoveIt 2 move_group for IK and Path Planning
    moveit_config_dir = os.path.join(pkg_share, 'config', 'moveit')
    
    # Load kinematics config
    kinematics_yaml_path = os.path.join(moveit_config_dir, 'kinematics.yaml')
    kinematics_config = {}
    try:
        if os.path.exists(kinematics_yaml_path):
            with open(kinematics_yaml_path, 'r') as f:
                kinematics_config = yaml.safe_load(f)
    except Exception:
        pass
    
    # Load OMPL planning config  
    ompl_planning_yaml_path = os.path.join(moveit_config_dir, 'ompl_planning.yaml')
    ompl_config = {}
    try:
        if os.path.exists(ompl_planning_yaml_path):
            with open(ompl_planning_yaml_path, 'r') as f:
                ompl_config = yaml.safe_load(f)
    except Exception:
        pass
    
    # Try to read SRDF
    srdf_path = os.path.join(moveit_config_dir, 'ur5e.srdf')
    robot_description_semantic = ''
    try:
        if os.path.exists(srdf_path):
            with open(srdf_path, 'r') as f:
                robot_description_semantic = f.read()
    except Exception:
        pass
    
    robot_description_semantic_param = {'robot_description_semantic': robot_description_semantic}
    
    moveit_config = {
        'robot_description_kinematics': kinematics_config,
        'robot_description_planning': ompl_config,
        'planning_scene_monitor_options': {
            'joint_state_topic': '/joint_states',
            'publish_planning_scene': True,
            'publish_geometry_updates': True,
            'publish_state_updates': True,
            'publish_transforms_updates': True,
        },
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'default_planning_pipeline': 'ompl',
            'start_state_max_bounds_error': 0.1,
        },
    }
    
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        output='screen',
        parameters=[
            {'robot_description': ParameterValue(Command(['cat ', urdf_path]), value_type=str)},
            robot_description_semantic_param,
            moveit_config,
            {'use_sim_time': False},
        ],
    )
    
    # 6. Foxglove Bridge
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': foxglove_port,
            'address': '0.0.0.0',
            'send_buffer_limit': 100000000,
            'use_compression': True,
            'asset_uri_allowlist': ['package://.*'],
        }],
    )
    
    return LaunchDescription([
        # Arguments
        robot_name_arg,
        websocket_port_arg,
        foxglove_port_arg,
        urdf_path_arg,
        
        # Nodes
        safety_watchdog_node,
        robot_state_publisher_node,
        static_tf_republisher,
        command_gateway_node,
        move_group_node,
        foxglove_bridge_node,
    ])
