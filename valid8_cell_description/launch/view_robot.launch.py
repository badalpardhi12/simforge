"""
Valid8 Cell Description Viewer

Launch file to visualize the Valid8 workcell in RViz with joint_state_publisher_gui.
Useful for testing the URDF/XACRO description before integrating with the control stack.

Usage:
    ros2 launch valid8_cell_description view_robot.launch.py
    ros2 launch valid8_cell_description view_robot.launch.py ur_type:=ur20
    ros2 launch valid8_cell_description view_robot.launch.py tf_prefix:=robot1_
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Arguments
    ur_type = LaunchConfiguration("ur_type")
    tf_prefix = LaunchConfiguration("tf_prefix")
    
    # Package paths
    description_package = FindPackageShare("valid8_cell_description")
    description_file = PathJoinSubstitution(
        [description_package, "urdf", "valid8_cell.urdf.xacro"]
    )
    rviz_config_file = PathJoinSubstitution(
        [description_package, "rviz", "view_robot.rviz"]
    )
    
    # Robot description (processed xacro)
    robot_description = ParameterValue(
        Command([
            "xacro ", description_file,
            " ur_type:=", ur_type,
            " tf_prefix:=", tf_prefix,
        ]),
        value_type=str
    )
    
    # Robot State Publisher - publishes TF from robot_description
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
    )
    
    # Joint State Publisher GUI - provides sliders to move joints
    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
    )
    
    # RViz - visualization
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
    )
    
    # Launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            description="Type/series of UR robot to use.",
            choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur20", "ur30"],
            default_value="ur5e",
        ),
        DeclareLaunchArgument(
            "tf_prefix",
            description="Prefix for all robot TF frames (e.g., 'ur5e_' or 'robot1_').",
            default_value="",
        ),
    ]
    
    return LaunchDescription(
        declared_arguments
        + [
            joint_state_publisher_gui_node,
            robot_state_publisher_node,
            rviz_node,
        ]
    )
