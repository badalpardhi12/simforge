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
    ExecuteProcess,
    TimerAction,
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
    reverse_ip = LaunchConfiguration("reverse_ip")
    
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
    
    # Headless mode network configuration
    declared_arguments.append(
        DeclareLaunchArgument(
            "reverse_ip",
            default_value="192.168.1.12",
            description="IP address of the ROS host machine for headless mode. "
                       "The robot URScript connects back to this IP. "
                       "Must be the actual IP, not 0.0.0.0.",
        )
    )
    
    # ========== UR Robot Driver (custom workcell description) ==========
    # Use our custom workcell description which includes:
    # - Shop floor, table, face fixture
    # - Robot positioned ON the table (not at world origin)
    # - iPhone tool attached to wrist_3_link
    # 
    # This properly integrates following the UR custom workcell tutorial:
    # https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/doc/ur_tutorials/
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
            
            # Use our CUSTOM workcell description with robot on table
            # This includes the full TF tree: world -> shop_floor -> robot_mount -> robot
            "description_package": "valid8_cell_description",
            "description_file": "valid8_cell.urdf.xacro",
            
            # Disable UR driver's RViz (we launch our own if needed)
            "launch_rviz": "false",
            
            # Use scaled trajectory controller
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            
            # Headless mode - no URCap required
            "headless_mode": "true",
            
            # Activate the joint controller on startup
            "activate_joint_controller": "true",
            
            # Reverse IP for headless mode - robot connects back to this address
            "reverse_ip": reverse_ip,
            
            # Mock hardware for simulation
            "use_fake_hardware": use_mock_hardware,
        }.items(),
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
    
    # ========== Headless Mode Activation ==========
    # In headless mode, the robot_program_running starts as false, which causes
    # controller_stopper to deactivate controllers. We need to:
    # 1. Call resend_robot_program to start the URScript on the robot
    # 2. Wait for robot_program_running to become true
    # 3. controller_stopper will then auto-activate the scaled_joint_trajectory_controller
    # 
    # CRITICAL ORDER: resend_robot_program FIRST, then wait. The controller_stopper
    # node (from ur_robot_driver) watches robot_program_running and automatically
    # activates/deactivates controllers. If we manually activate the controller
    # while robot_program_running is false, controller_stopper immediately
    # deactivates it again (race condition).
    headless_activation = TimerAction(
        period=15.0,  # Wait 15 seconds for UR driver to be fully ready
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash", "-c",
                    ". /ros2_ws/install/setup.bash && "
                    "echo '[headless_keepalive] Starting headless mode keepalive monitor' && "
                    "while true; do "
                    "  RUNNING=$(ros2 topic echo /io_and_status_controller/robot_program_running --once 2>/dev/null | grep -c 'true' || true); "
                    "  if [ \"$RUNNING\" = \"0\" ]; then "
                    "    echo '[headless_keepalive] robot_program_running is false, resending program...'; "
                    "    ros2 service call /io_and_status_controller/resend_robot_program std_srvs/srv/Trigger 2>/dev/null && "
                    "    echo '[headless_keepalive] Robot program resent, waiting for controller_stopper to activate controllers...'; "
                    "    sleep 3; "
                    "    CTRL_STATE=$(ros2 control list_controllers 2>/dev/null | grep scaled_joint_trajectory_controller | grep -c active || true); "
                    "    if [ \"$CTRL_STATE\" = \"0\" ]; then "
                    "      echo '[headless_keepalive] Controller still inactive, forcing activation...'; "
                    "      ros2 control set_controller_state scaled_joint_trajectory_controller active 2>/dev/null || true; "
                    "    fi; "
                    "  fi; "
                    "  sleep 5; "
                    "done"
                ],
                output="screen",
            ),
        ],
    )

    return LaunchDescription(
        declared_arguments
        + [
            # Core robot control (includes robot_state_publisher with full workcell TF tree)
            # Robot is properly positioned on table via valid8_cell.urdf.xacro
            ur_control_launch,
            
            # Motion planning
            moveit_launch,
            
            # Visualization
            foxglove_bridge_node,
            rviz_node,
            
            # Application layer
            command_gateway_node,
            
            # Headless mode activation (delayed to ensure driver is ready)
            headless_activation,
        ]
    )
