"""
Valid8 Cell Robot State Publisher Launch File

This launch file starts the robot_state_publisher with the controlled URDF.
It is used by the UR Robot Driver via the description_launchfile parameter.

The robot_state_publisher:
- Publishes the robot_description to the /robot_description topic
- Subscribes to /joint_states and publishes TF transforms
- Handles the complete workcell including robot, table, and face

Usage:
    ros2 launch valid8_cell_control rsp.launch.py
    ros2 launch valid8_cell_control rsp.launch.py ur_type:=ur20
    ros2 launch valid8_cell_control rsp.launch.py use_mock_hardware:=true

Based on Universal Robots ROS 2 custom workcell tutorial:
https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/doc/ur_tutorials/my_robot_cell/doc/start_ur_driver.html
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get launch configurations
    ur_type = LaunchConfiguration("ur_type")
    tf_prefix = LaunchConfiguration("tf_prefix")
    robot_ip = LaunchConfiguration("robot_ip")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    mock_sensor_commands = LaunchConfiguration("mock_sensor_commands")
    headless_mode = LaunchConfiguration("headless_mode")
    
    # Build robot description from controlled XACRO
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("valid8_cell_control"),
                    "urdf",
                    "valid8_cell_controlled.urdf.xacro",
                ]
            ),
            " ",
            "robot_ip:=", robot_ip,
            " ",
            "ur_type:=", ur_type,
            " ",
            "tf_prefix:=", tf_prefix,
            " ",
            "use_mock_hardware:=", use_mock_hardware,
            " ",
            "mock_sensor_commands:=", mock_sensor_commands,
            " ",
            "headless_mode:=", headless_mode,
        ]
    )
    robot_description = {"robot_description": robot_description_content}
    
    # Declare arguments
    declared_arguments = []
    
    # UR specific arguments
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
            description="Prefix for all robot TF frames.",
            default_value="",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            description="IP address of the robot.",
            default_value="192.168.1.9",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="false",
            description="Start robot with mock hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "mock_sensor_commands",
            default_value="false",
            description="Enable mock command interfaces for sensors used for simple simulations.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "headless_mode",
            default_value="true",
            description="Enable headless mode for robot control without URCap.",
        )
    )
    
    return LaunchDescription(
        declared_arguments
        + [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="both",
                parameters=[robot_description],
            ),
        ]
    )
