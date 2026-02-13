"""
Robot State Publisher Launch File — Face Robot UR20

Launches robot_state_publisher with the controlled URDF for a single UR20.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    headless_mode = LaunchConfiguration("headless_mode")

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("face_robot_ur20_control"),
                    "urdf",
                    "face_robot_ur20_controlled.urdf.xacro",
                ]
            ),
            " ",
            "robot_ip:=", robot_ip,
            " ",
            "use_fake_hardware:=", use_fake_hardware,
            " ",
            "fake_sensor_commands:=", fake_sensor_commands,
            " ",
            "headless_mode:=", headless_mode,
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    declared_arguments = [
        DeclareLaunchArgument("robot_ip", default_value="0.0.0.0",
                              description="IP address of the UR20 robot."),
        DeclareLaunchArgument("use_fake_hardware", default_value="false",
                              description="Start with fake/simulated hardware."),
        DeclareLaunchArgument("fake_sensor_commands", default_value="false",
                              description="Enable fake sensor commands."),
        DeclareLaunchArgument("headless_mode", default_value="false",
                              description="Enable headless mode."),
    ]

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
