"""
View Robot Launch File

Launch file to visualize the dual robot cell in RViz with joint_state_publisher_gui.
"""
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    description_package = FindPackageShare("valid8_dual_cell_description")
    description_file = PathJoinSubstitution(
        [description_package, "urdf", "valid8_dual_cell.urdf.xacro"]
    )
    rvizconfig_file = PathJoinSubstitution([description_package, "rviz", "urdf.rviz"])

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                description_file,
                " ",
                "nakul_ur_type:=",
                "ur5e",
                " ",
                "sahadev_ur_type:=",
                "ur5e",
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rvizconfig_file],
    )

    return LaunchDescription(
        [joint_state_publisher_gui_node, robot_state_publisher_node, rviz_node]
    )
