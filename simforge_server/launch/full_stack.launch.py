"""
Full Stack Launch File

Launches all Simforge server components:
- Safety Watchdog (first)
- Robot Control
- Robot State Publisher (for TF and URDF visualization)
- Command Gateway
- Perception
- VLA Inference
- Orchestrator
- Foxglove Bridge

Usage:
    ros2 launch simforge_server full_stack.launch.py
    ros2 launch simforge_server full_stack.launch.py robot_ip:=192.168.1.9
    ros2 launch simforge_server full_stack.launch.py simulation_mode:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for full stack."""
    
    # Get package share directory for asset paths
    pkg_share = get_package_share_directory('simforge_server')
    
    # Declare launch arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.9',
        description='IP address of the UR robot'
    )
    
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='nakul_ur5e',
        description='Name of the robot'
    )
    
    simulation_mode_arg = DeclareLaunchArgument(
        'simulation_mode',
        default_value='false',
        description='Run in simulation mode without real hardware'
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
    
    use_vla_arg = DeclareLaunchArgument(
        'use_vla',
        default_value='true',
        description='Enable VLA inference node'
    )
    
    use_perception_arg = DeclareLaunchArgument(
        'use_perception',
        default_value='true',
        description='Enable perception node'
    )
    
    # Default URDF path - combined environment URDF with robot, table, and face
    default_urdf = os.path.join(pkg_share, 'assets', 'valid8_environment.urdf')
    
    urdf_path_arg = DeclareLaunchArgument(
        'urdf_path',
        default_value=default_urdf,
        description='Path to combined environment URDF file (robot + table + face)'
    )
    
    # Get launch configurations
    robot_ip = LaunchConfiguration('robot_ip')
    robot_name = LaunchConfiguration('robot_name')
    simulation_mode = LaunchConfiguration('simulation_mode')
    websocket_port = LaunchConfiguration('websocket_port')
    foxglove_port = LaunchConfiguration('foxglove_port')
    urdf_path = LaunchConfiguration('urdf_path')
    use_vla = LaunchConfiguration('use_vla')
    use_perception = LaunchConfiguration('use_perception')
    
    # === Nodes ===
    
    # 1. Safety Watchdog (CRITICAL - starts first)
    safety_watchdog_node = Node(
        package='simforge_server',
        executable='safety_watchdog_node.py',
        name='safety_watchdog',
        output='screen',
        parameters=[{
            'heartbeat_timeout_sec': 2.0,  # 2 seconds - tolerant for network variations
            'max_consecutive_misses': 5,   # Allow more misses before protective stop
            'check_frequency_hz': 10.0,    # 10Hz monitoring - sufficient for safety
            'enable_force_monitoring': True,
            'max_tcp_force_n': 100.0,
            'max_tcp_torque_nm': 10.0,
        }],
    )
    
    # 2. Robot Control (v2 - uses Dashboard + Secondary/Realtime interfaces)
    robot_control_node = Node(
        package='simforge_server',
        executable='robot_control_node_v2.py',
        name='robot_control',
        output='screen',
        parameters=[{
            'robot_ip': robot_ip,
            'robot_name': robot_name,
            'state_publish_rate': 50.0,
            'simulation_mode': simulation_mode,
        }],
    )
    
    # 2b. Robot State Publisher (publishes TF transforms from joint_states + combined URDF)
    # This single publisher handles the robot, table, and face - all in one URDF
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
    
    # 2c. Static Transform Republisher (republishes /tf_static to /tf for Foxglove)
    # Foxglove websocket bridge doesn't handle transient local QoS well
    static_tf_republisher = Node(
        package='simforge_server',
        executable='static_transform_republisher.py',
        name='static_tf_republisher',
        output='screen',
        parameters=[{
            'publish_rate': 5.0,  # 5 Hz republish rate
        }],
    )
    
    # 3. Command Gateway
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
    
    # 4. Perception Node
    perception_node = Node(
        package='simforge_server',
        executable='perception_node.py',
        name='perception',
        output='screen',
        parameters=[{
            'camera_topic_prefix': '/camera',
            'use_simulation': simulation_mode,
            'voxel_size': 0.05,
            'publish_rate': 10.0,
        }],
        condition=IfCondition(use_perception),
    )
    
    # 5. VLA Inference Node
    vla_inference_node = Node(
        package='simforge_server',
        executable='vla_inference_node.py',
        name='vla_inference',
        output='screen',
        parameters=[{
            'model_name': 'openvla/openvla-7b',
            'model_path': '/models/openvla',
            'use_tensorrt': False,
            'precision': 'fp16',
            'simulation_mode': simulation_mode,
        }],
        condition=IfCondition(use_vla),
    )
    
    # 6. Orchestrator Node
    orchestrator_node = Node(
        package='simforge_server',
        executable='orchestrator_node.py',
        name='orchestrator',
        output='screen',
        parameters=[{
            'default_robot': robot_name,
            'camera_topic': '/camera/color/image_raw',
            'max_velocity_scale': 0.5,
            'collision_check_enabled': True,
            'max_replans': 3,
        }],
    )
    
    # 7. Foxglove Bridge
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': foxglove_port,
            'address': '0.0.0.0',
            'send_buffer_limit': 100000000,  # 100MB for point clouds
            'use_compression': True,
            'asset_uri_allowlist': ['package://.*'],  # Allow fetching package:// assets
        }],
    )
    
    return LaunchDescription([
        # Arguments
        robot_ip_arg,
        robot_name_arg,
        simulation_mode_arg,
        websocket_port_arg,
        foxglove_port_arg,
        use_vla_arg,
        use_perception_arg,
        urdf_path_arg,
        
        # Nodes (in dependency order)
        safety_watchdog_node,
        robot_control_node,
        robot_state_publisher_node,  # Publishes TF from joint_states + combined URDF
        static_tf_republisher,       # Republishes static TFs to /tf for Foxglove
        command_gateway_node,
        perception_node,
        vla_inference_node,
        orchestrator_node,
        foxglove_bridge_node,
    ])
