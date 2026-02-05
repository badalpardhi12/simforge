"""
Valid8 Cell Start Robot Launch File

This launch file starts the complete robot control stack including:
- UR Robot Driver (via ur_control.launch.py)
- Custom robot_state_publisher with Valid8 cell description
- Controllers (scaled_joint_trajectory_controller)

The key insight is passing our custom description_launchfile to the UR driver,
which allows us to include the complete workcell (robot, table, face) in TF.

Usage:
    # Mock hardware (simulation)
    ros2 launch valid8_cell_control start_robot.launch.py use_mock_hardware:=true
    
    # Real robot
    ros2 launch valid8_cell_control start_robot.launch.py robot_ip:=192.168.1.9
    
    # Different robot type
    ros2 launch valid8_cell_control start_robot.launch.py ur_type:=ur20 use_mock_hardware:=true

Based on Universal Robots ROS 2 custom workcell tutorial:
https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/doc/ur_tutorials/my_robot_cell/doc/start_ur_driver.html
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get launch configurations
    ur_type = LaunchConfiguration("ur_type")
    tf_prefix = LaunchConfiguration("tf_prefix")
    robot_ip = LaunchConfiguration("robot_ip")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    launch_rviz = LaunchConfiguration("launch_rviz")
    
    # Declare arguments
    declared_arguments = []
    
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            description="Type/series of UR robot.",
            choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20", "ur30"],
            default_value="ur5e",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tf_prefix",
            description="Prefix for all robot TF frames (e.g., 'ur5e_' for multi-robot).",
            default_value="",
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
            "use_mock_hardware",
            default_value="false",
            description="Use mock hardware for simulation without real robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz for visualization.",
        )
    )
    
    # Include UR Robot Driver launch
    # The key is passing our custom description_launchfile
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
            # IMPORTANT: Use the tf_prefix argument to namespace robot frames
            "tf_prefix": tf_prefix,
            
            # Use our custom description launch file
            # This is what makes the whole workcell visible in TF
            "description_launchfile": PathJoinSubstitution([
                FindPackageShare("valid8_cell_control"),
                "launch",
                "rsp.launch.py",
            ]),
            
            # RViz configuration
            "launch_rviz": "false",  # We launch our own RViz if needed
            "rviz_config_file": PathJoinSubstitution([
                FindPackageShare("valid8_cell_description"),
                "rviz",
                "view_robot.rviz",
            ]),
            
            # Controller configuration
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            
            # Headless mode - no URCap required
            "headless_mode": "true",
            
            # Mock hardware for simulation
            "use_mock_hardware": use_mock_hardware,
        }.items(),
    )
    
    # Optional RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(launch_rviz),
        arguments=[
            "-d",
            PathJoinSubstitution([
                FindPackageShare("valid8_cell_description"),
                "rviz",
                "view_robot.rviz",
            ]),
        ],
    )
    
    return LaunchDescription(
        declared_arguments
        + [
            ur_control_launch,
            rviz_node,
        ]
    )
