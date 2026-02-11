"""
Robot State Publisher Launch File

Launches the robot_state_publisher with the controlled URDF for dual robots.
Based on Universal Robots ROS2 Dual Robot Tutorial.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Robot IPs
    nakul_robot_ip = LaunchConfiguration("nakul_robot_ip")
    sahadev_robot_ip = LaunchConfiguration("sahadev_robot_ip")

    # Fake hardware settings (for simulation)
    nakul_use_fake_hardware = LaunchConfiguration("nakul_use_fake_hardware")
    nakul_fake_sensor_commands = LaunchConfiguration("nakul_fake_sensor_commands")
    sahadev_use_fake_hardware = LaunchConfiguration("sahadev_use_fake_hardware")
    sahadev_fake_sensor_commands = LaunchConfiguration("sahadev_fake_sensor_commands")

    # Other settings
    headless_mode = LaunchConfiguration("headless_mode")

    # Build robot description from xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("valid8_dual_cell_control"),
                    "urdf",
                    "valid8_dual_cell_controlled.urdf.xacro",
                ]
            ),
            " ",
            "nakul_robot_ip:=", nakul_robot_ip,
            " ",
            "sahadev_robot_ip:=", sahadev_robot_ip,
            " ",
            "nakul_use_fake_hardware:=", nakul_use_fake_hardware,
            " ",
            "sahadev_use_fake_hardware:=", sahadev_use_fake_hardware,
            " ",
            "nakul_fake_sensor_commands:=", nakul_fake_sensor_commands,
            " ",
            "sahadev_fake_sensor_commands:=", sahadev_fake_sensor_commands,
            " ",
            "headless_mode:=", headless_mode,
        ]
    )
    # Wrap in ParameterValue to ensure it's treated as a string
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    declared_arguments = []
    
    # Robot IP arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "nakul_robot_ip",
            default_value="192.168.1.9",
            description="IP address of nakul robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "sahadev_robot_ip",
            default_value="192.168.1.16",
            description="IP address of sahadev robot.",
        )
    )
    
    # Fake hardware arguments (for simulation)
    declared_arguments.append(
        DeclareLaunchArgument(
            "nakul_use_fake_hardware",
            default_value="false",
            description="Start nakul with fake/simulated hardware.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "sahadev_use_fake_hardware",
            default_value="false",
            description="Start sahadev with fake/simulated hardware.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "nakul_fake_sensor_commands",
            default_value="false",
            description="Enable fake sensor commands for nakul.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "sahadev_fake_sensor_commands",
            default_value="false",
            description="Enable fake sensor commands for sahadev.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "headless_mode",
            default_value="false",
            description="Enable headless mode for robot control.",
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
