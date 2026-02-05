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
from launch.substitutions import LaunchConfiguration, PythonExpression, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
    
    use_ur_driver_arg = DeclareLaunchArgument(
        'use_ur_driver',
        default_value='true',
        description='Enable UR Robot Driver for real robot trajectory control'
    )
    
    ur_type_arg = DeclareLaunchArgument(
        'ur_type',
        default_value='ur5e',
        description='Type of UR robot (ur3e, ur5e, ur10e, ur16e, ur20, ur30)'
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
    use_ur_driver = LaunchConfiguration('use_ur_driver')
    ur_type = LaunchConfiguration('ur_type')
    
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
    # This is used when UR driver is NOT enabled (fallback mode)
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
        condition=UnlessCondition(use_ur_driver),
    )
    
    # 2a. UR Robot Driver (preferred for real robot trajectory control)
    # This launches the official Universal Robots ROS 2 driver which provides:
    # - scaled_joint_trajectory_controller for proper trajectory execution
    # - Joint state publishing at high frequency
    # - Dashboard communication
    # - headless_mode for programmatic control without URCap
    # NOTE: We use tf_prefix='ur_' to avoid TF conflicts with our valid8_environment.urdf
    # The UR driver publishes world -> base_link at origin, but we need robot on the table
    ur_robot_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory('ur_robot_driver'),
                'launch',
                'ur_control.launch.py'
            ])
        ]),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': robot_ip,
            'headless_mode': 'true',  # No URCap needed
            'launch_rviz': 'false',   # We use Foxglove instead
            'initial_joint_controller': 'scaled_joint_trajectory_controller',
            'tf_prefix': '',  # Keep empty to use standard frame names
        }.items(),
        condition=IfCondition(use_ur_driver),
    )
    
    # 2b. Robot State Publisher (publishes TF transforms from joint_states + combined URDF)
    # This single publisher handles the robot, table, and face - all in one URDF
    # Only used when NOT using UR driver (UR driver has its own robot_state_publisher)
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
        condition=UnlessCondition(use_ur_driver),
    )
    
    # 2b-alt. When using UR driver, we need static transforms to position environment elements
    # relative to the UR driver's TF tree. The UR driver publishes world -> base_link at origin.
    # We connect our environment elements (table, face) to the UR driver's base_link frame
    # using the inverse of the robot's offset from valid8_environment.urdf
    
    # Static transform: base_link -> table_link 
    # Robot is at [-0.6758, 0, 1.03] relative to table origin in valid8_environment.urdf
    # So table is at [0.6758, 0, -1.03] relative to base_link, then add table's z offset (1.0m)
    # Table position relative to base_link: [0.6758, 0, 1.0 - 1.03] = [0.6758, 0, -0.03]
    # Also need to account for robot's -90° yaw: rotate positions
    # With -90° yaw (robot facing -Y), X->Y and Y->-X
    # So table at world [0,0,1] with robot at world [-0.6758, 0, 1.03] facing -Y:
    # In robot's base_link frame: table is at [0 - (-0.6758), 0 - 0, 1.0 - 1.03] = [0.6758, 0, -0.03]
    # But with the yaw rotation... let me just use world as the parent
    
    # Actually, the simplest fix is to NOT fight the UR driver.
    # The UR driver puts the robot at origin. Let's position our environment relative to the robot.
    
    # Table position: If robot base is at world origin, and in our setup robot was at [-0.6758, 0, 1.03],
    # then table (which was at [0, 0, 1.0]) is at [0.6758, 0, -0.03] relative to robot
    # But we need to transform this by the robot's yaw (-90°):
    # After -90° yaw: x' = y, y' = -x -> [0, -0.6758, -0.03]? No that's wrong.
    # Let's just use world frame from UR driver and position elements there
    
    # UR driver's world frame is at origin. We want:
    # - Robot base_link at origin (UR driver handles this)
    # - Table at position relative to where robot was in our setup
    # 
    # In valid8_environment.urdf:
    # - table_link at [0, 0, 1.0] in world
    # - base_link at [-0.6758, 0, 1.03] with -90° yaw in world
    # - face_link at [0.1742, 0, 1.6] with 90° yaw in world
    #
    # When UR driver puts robot at origin (0,0,0) with 0° yaw:
    # The robot's base_link is at the table surface height (1.03m above floor).
    # We need to position environment elements relative to this new origin.
    #
    # Original relative positions (robot at [-0.6758, 0, 1.03] with -90° yaw):
    # - Table center at [0, 0, 1.0] -> offset from robot: [0.6758, 0, -0.03]
    # - Face at [0.1742, 0, 1.6] -> offset from robot: [0.85, 0, 0.57]
    #
    # With robot yaw at -90°, the robot was facing -Y direction.
    # The table is "to the right" (+X) of robot and "behind" (+Y) relative to robot facing.
    # Face is "in front" and "to the right" (+X) of robot.
    #
    # When UR driver puts robot facing +X (0° yaw):
    # - Table stays at [0.6758, 0, -0.03] (directly to the right of robot base)
    # - Face at [0.85, 0, 0.57] with 90° yaw
    #
    # But we also need to account for the robot's original -90° yaw.
    # The face was positioned for the robot facing -Y. With robot now facing +X,
    # the face position needs to rotate by -90° around Z axis.
    # Rotation: x' = x*cos(-90°) - y*sin(-90°) = y, y' = x*sin(-90°) + y*cos(-90°) = -x
    # Face [0.85, 0] becomes [0, -0.85] - but this seems wrong for in front of robot
    #
    # Actually, let's keep it simple and position things for the UR driver:
    # - Robot base_link at origin, table is BELOW (-Z) by ~1.03m
    # - Face is in front of the robot (+X direction), at a reasonable height
    
    # Table: Positioned below and slightly in front of the robot
    # The table surface should be at Z=0 (where robot base sits)
    # Table visual origin is at its center, so we put table_link at z=-0.03 (table surface ~3cm below robot base)
    optical_table_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='optical_table_static_tf',
        arguments=[
            '--x', '0.3', '--y', '0', '--z', '-0.03',
            '--frame-id', 'base_link', '--child-frame-id', 'table_link'
        ],
        output='screen',
        condition=IfCondition(use_ur_driver),
    )
    
    # Face: Positioned in front of the robot's typical working area
    # About 0.5m in front (+X) and at head height relative to table (~0.6m above robot base)
    # Face oriented to face the robot (yaw = 180° = 3.14159)
    face_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='face_static_tf',
        arguments=[
            '--x', '0.85', '--y', '0', '--z', '0.57',
            '--roll', '0', '--pitch', '0', '--yaw', '3.14159',
            '--frame-id', 'base_link', '--child-frame-id', 'face_link'
        ],
        output='screen',
        condition=IfCondition(use_ur_driver),
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
    
    # 7. MoveIt 2 move_group for IK and Path Planning
    # Load configs
    import yaml
    moveit_config_dir = os.path.join(pkg_share, 'config', 'moveit')
    print(f"[MoveIt Config] Looking for configs in: {moveit_config_dir}")
    print(f"[MoveIt Config] Directory exists: {os.path.exists(moveit_config_dir)}")
    
    # Load kinematics config
    kinematics_yaml_path = os.path.join(moveit_config_dir, 'kinematics.yaml')
    print(f"[MoveIt Config] kinematics.yaml exists: {os.path.exists(kinematics_yaml_path)}")
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
    
    # Load joint limits config
    joint_limits_yaml_path = os.path.join(moveit_config_dir, 'joint_limits.yaml')
    joint_limits_config = {}
    try:
        if os.path.exists(joint_limits_yaml_path):
            with open(joint_limits_yaml_path, 'r') as f:
                joint_limits_config = yaml.safe_load(f)
    except Exception:
        pass
    
    # Try to read SRDF
    srdf_path = os.path.join(moveit_config_dir, 'ur5e.srdf')
    print(f"[MoveIt Config] SRDF path: {srdf_path}")
    print(f"[MoveIt Config] SRDF exists: {os.path.exists(srdf_path)}")
    robot_description_semantic = ''
    try:
        if os.path.exists(srdf_path):
            with open(srdf_path, 'r') as f:
                robot_description_semantic = f.read()
            print(f"[MoveIt Config] SRDF loaded, length={len(robot_description_semantic)} chars")
        else:
            print(f"[MoveIt Config] ERROR: SRDF file not found at {srdf_path}")
            # List available files in directory
            if os.path.exists(moveit_config_dir):
                print(f"[MoveIt Config] Files in {moveit_config_dir}: {os.listdir(moveit_config_dir)}")
    except Exception as e:
        print(f"[MoveIt Config] ERROR reading SRDF: {e}")
    
    # MoveIt configuration - split into separate dicts as MoveIt expects
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
            'capabilities': '',
            'disable_capabilities': '',
            'publish_robot_description': True,
            'publish_robot_description_semantic': True,
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
    
    # 8. Foxglove Bridge
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
        use_ur_driver_arg,
        ur_type_arg,
        urdf_path_arg,
        
        # Nodes (in dependency order)
        safety_watchdog_node,
        robot_control_node,           # Only when use_ur_driver:=false
        ur_robot_driver_launch,       # Only when use_ur_driver:=true
        robot_state_publisher_node,   # Publishes TF from joint_states + combined URDF (only when NOT using UR driver)
        optical_table_static_tf,      # Optical table frame (only when using UR driver)
        face_static_tf,               # Face frame (only when using UR driver)
        static_tf_republisher,        # Republishes static TFs to /tf for Foxglove
        command_gateway_node,
        perception_node,
        vla_inference_node,
        orchestrator_node,
        move_group_node,              # MoveIt 2 for IK and path planning
        foxglove_bridge_node,
    ])
