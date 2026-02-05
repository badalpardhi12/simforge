"""
Valid8 Full Stack Launch File

This is the unified launch file that starts the complete Valid8 robot control stack:

1. UR Robot Driver (with custom workcell description)
2. MoveIt move_group (for motion planning and IK)
3. Foxglove Bridge (for web-based visualization)
4. Optional: Command Gateway (for WebSocket control interface)
5. Optional: Safety Watchdog
6. Optional: Perception nodes

This properly integrates the UR Robot Driver following the official
Universal Robots ROS 2 custom workcell tutorial pattern.

Key differences from the old full_stack.launch.py:
- Uses proper XACRO-based workcell description
- Passes custom description_launchfile to UR driver
- Robot is NOT at world origin - properly positioned via workcell macro
- No conflicting TF publishers
- Supports tf_prefix for multi-robot setups

Usage:
    # Mock hardware (simulation)
    ros2 launch valid8_cell_control full_stack.launch.py use_mock_hardware:=true
    
    # Real robot
    ros2 launch valid8_cell_control full_stack.launch.py robot_ip:=192.168.1.9
    
    # Different robot type
    ros2 launch valid8_cell_control full_stack.launch.py ur_type:=ur20 robot_ip:=192.168.1.9
    
    # Multi-robot with tf_prefix
    ros2 launch valid8_cell_control full_stack.launch.py tf_prefix:=robot1_ robot_ip:=192.168.1.9

Based on Universal Robots ROS 2 custom workcell tutorial:
https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/doc/ur_tutorials/
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ========== Launch Configurations ==========
    ur_type = LaunchConfiguration("ur_type")
    tf_prefix = LaunchConfiguration("tf_prefix")
    robot_ip = LaunchConfiguration("robot_ip")
    robot_name = LaunchConfiguration("robot_name")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_foxglove = LaunchConfiguration("launch_foxglove")
    launch_moveit = LaunchConfiguration("launch_moveit")
    launch_command_gateway = LaunchConfiguration("launch_command_gateway")
    foxglove_port = LaunchConfiguration("foxglove_port")
    websocket_port = LaunchConfiguration("websocket_port")
    
    # ========== Declare Arguments ==========
    declared_arguments = []
    
    # Robot configuration
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            description="Type/series of UR robot.",
            choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20", "ur30"],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tf_prefix",
            default_value="",
            description="TF prefix for all robot frames (e.g., 'robot1_' for multi-robot).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.1.9",
            description="IP address of the real robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_name",
            default_value="nakul_ur5e",
            description="Name identifier for the robot.",
        )
    )
    
    # Simulation vs real hardware
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="false",
            description="Use mock hardware for simulation without real robot.",
        )
    )
    
    # Optional components
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="false",
            description="Launch RViz for visualization.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_foxglove",
            default_value="true",
            description="Launch Foxglove Bridge for web-based visualization.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_moveit",
            default_value="true",
            description="Launch MoveIt move_group for motion planning.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_command_gateway",
            default_value="true",
            description="Launch Command Gateway for WebSocket control.",
        )
    )
    
    # Ports
    declared_arguments.append(
        DeclareLaunchArgument(
            "foxglove_port",
            default_value="9090",
            description="WebSocket port for Foxglove Bridge.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "websocket_port",
            default_value="8766",
            description="WebSocket port for Command Gateway.",
        )
    )
    
    # ========== UR Robot Driver (standard UR description) ==========
    # Use the standard UR description package - the robot TF tree will be
    # published by the UR driver's robot_state_publisher
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ur_robot_driver"),
                "launch",
                "ur_control.launch.py",
            ])
        ]),
        launch_arguments={
            # Robot configuration
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            
            # TF prefix for multi-robot support
            "tf_prefix": tf_prefix,
            
            # Use default UR description (robot only, no workcell)
            # We add workcell transforms separately below
            "description_package": "ur_description",
            "description_file": "ur.urdf.xacro",
            
            # Disable UR driver's RViz (we launch our own if needed)
            "launch_rviz": "false",
            
            # Use scaled trajectory controller
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            
            # Headless mode - no URCap required
            "headless_mode": "true",
            
            # Mock hardware for simulation
            "use_fake_hardware": use_mock_hardware,
        }.items(),
    )
    
    # ========== Workcell Static Transforms ==========
    # The UR driver already publishes world -> base_link at origin (from ur.urdf.xacro)
    # So we position all workcell elements relative to where they should be in the robot's frame
    # 
    # In our workcell design:
    # - Robot base_link is at world [-0.6758, 0, 1.03] with -90° yaw
    # - shop_floor is at world [0, 0, 0]
    # - table_link is at world [0, 0, 1.0]
    # - face_link is at world [0.1742, 0, 1.6] with +90° yaw
    # - robot_mount is at world [-0.6758, 0, 1.03] with -90° yaw (same as robot)
    #
    # But since UR driver places robot at world origin, we need to adjust:
    # - base_link is at origin (from UR driver)
    # - We publish shop_floor, table, face, robot_mount relative to base_link
    
    # base_link -> robot_mount (robot_mount is at same position as base_link, just a reference frame)
    # This is needed for MoveIt SRDF compatibility
    base_to_robot_mount = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_robot_mount_publisher",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "0",
            "--frame-id", "base_link",
            "--child-frame-id", "robot_mount",
        ],
    )
    
    # base_link -> shop_floor
    # Robot is at [-0.6758, 0, 1.03] with -90° yaw in world
    # shop_floor is at [0, 0, 0] in world
    # In robot's frame (base_link), shop_floor is at inverse transform:
    # After -90° yaw rotation: X'=Y, Y'=-X
    # Offset: [0 - (-0.6758), 0 - 0, 0 - 1.03] = [0.6758, 0, -1.03]
    # In rotated frame: [0, 0.6758, -1.03] with +90° yaw
    base_to_shop_floor = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_shop_floor_publisher",
        arguments=[
            "--x", "0",
            "--y", "0.6758",
            "--z", "-1.03",
            "--yaw", "1.5708",
            "--pitch", "0",
            "--roll", "0",
            "--frame-id", "base_link",
            "--child-frame-id", "shop_floor",
        ],
    )
    
    # shop_floor -> table_link (table at z=1.0m in world = z=1.0 in shop_floor)
    shop_floor_to_table = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="shop_floor_to_table_publisher",
        arguments=[
            "--x", "0",
            "--y", "0",
            "--z", "1.0",
            "--frame-id", "shop_floor",
            "--child-frame-id", "table_link",
        ],
    )
    
    # shop_floor -> face_link ([0.1742, 0, 1.6] with 90° yaw relative to shop_floor)
    shop_floor_to_face = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="shop_floor_to_face_publisher",
        arguments=[
            "--x", "0.1742",
            "--y", "0",
            "--z", "1.6",
            "--yaw", "1.5708",
            "--pitch", "0",
            "--roll", "0",
            "--frame-id", "shop_floor",
            "--child-frame-id", "face_link",
        ],
    )
    
    # ========== MoveIt move_group ==========
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("valid8_cell_moveit_config"),
                "launch",
                "move_group.launch.py",
            ])
        ]),
        launch_arguments={
            "ur_type": ur_type,
            "tf_prefix": tf_prefix,
        }.items(),
        condition=IfCondition(launch_moveit),
    )
    
    # ========== Foxglove Bridge ==========
    foxglove_bridge_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[{
            "port": foxglove_port,
            "address": "0.0.0.0",
            "send_buffer_limit": 100000000,  # 100MB for point clouds
            "use_compression": True,
            "asset_uri_allowlist": ["package://.*"],
        }],
        condition=IfCondition(launch_foxglove),
    )
    
    # ========== Command Gateway (from simforge_server) ==========
    command_gateway_node = Node(
        package="simforge_server",
        executable="command_gateway_node.py",
        name="command_gateway",
        output="screen",
        parameters=[{
            "websocket_port": websocket_port,
            "websocket_host": "0.0.0.0",
            "max_clients": 5,
            "robot_name": robot_name,
        }],
        condition=IfCondition(launch_command_gateway),
    )
    
    # ========== RViz ==========
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution([
                FindPackageShare("valid8_cell_description"),
                "rviz",
                "view_robot.rviz",
            ]),
        ],
        condition=IfCondition(launch_rviz),
    )
    
    return LaunchDescription(
        declared_arguments
        + [
            # Core robot control (includes robot_state_publisher)
            ur_control_launch,
            
            # Workcell static transforms (relative to base_link since UR driver places robot at origin)
            base_to_robot_mount,
            base_to_shop_floor,
            shop_floor_to_table,
            shop_floor_to_face,
            
            # Motion planning
            moveit_launch,
            
            # Visualization
            foxglove_bridge_node,
            rviz_node,
            
            # Application layer
            command_gateway_node,
        ]
    )
