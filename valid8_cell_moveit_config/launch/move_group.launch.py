"""
MoveIt move_group Launch File for Valid8 Cell

Launches the MoveIt move_group node which provides:
- Motion planning (via OMPL)
- IK solving (via KDL or ur_kinematics)
- Trajectory execution (via scaled_joint_trajectory_controller)
- Planning scene management

This should be launched AFTER the robot driver (valid8_cell_control start_robot.launch.py).

Usage:
    ros2 launch valid8_cell_moveit_config move_group.launch.py
    ros2 launch valid8_cell_moveit_config move_group.launch.py ur_type:=ur20

Reference: https://moveit.picknik.ai/main/doc/concepts/move_group.html
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import os
import yaml


def load_yaml(package_name, file_path):
    """Load a yaml file from a package."""
    package_path = FindPackageShare(package_name)
    absolute_file_path = PathJoinSubstitution([package_path, file_path])
    
    # For now, we'll use the package prefix to get the actual path
    from ament_index_python.packages import get_package_share_directory
    package_dir = get_package_share_directory(package_name)
    full_path = os.path.join(package_dir, file_path)
    
    try:
        with open(full_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Warning: Could not load {full_path}: {e}")
        return {}


def generate_launch_description():
    # Launch configurations
    ur_type = LaunchConfiguration("ur_type")
    tf_prefix = LaunchConfiguration("tf_prefix")
    
    # Declare arguments
    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e",
            description="Type of UR robot.",
        ),
        DeclareLaunchArgument(
            "tf_prefix",
            default_value="",
            description="TF prefix for robot frames.",
        ),
    ]
    
    # Robot description (from description package)
    robot_description_content = Command([
        PathJoinSubstitution(["xacro"]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("valid8_cell_description"),
            "urdf",
            "valid8_cell.urdf.xacro",
        ]),
        " ur_type:=", ur_type,
        " tf_prefix:=", tf_prefix,
    ])
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}
    
    # SRDF (from moveit_config package)
    from ament_index_python.packages import get_package_share_directory
    moveit_config_dir = get_package_share_directory("valid8_cell_moveit_config")
    
    srdf_path = os.path.join(moveit_config_dir, "config", "valid8_cell.srdf")
    with open(srdf_path, 'r') as f:
        robot_description_semantic = {"robot_description_semantic": f.read()}
    
    # Load other MoveIt configs
    kinematics_yaml = load_yaml("valid8_cell_moveit_config", "config/kinematics.yaml")
    ompl_planning_yaml = load_yaml("valid8_cell_moveit_config", "config/ompl_planning.yaml")
    joint_limits_yaml = load_yaml("valid8_cell_moveit_config", "config/joint_limits.yaml")
    moveit_controllers_yaml = load_yaml("valid8_cell_moveit_config", "config/moveit_controllers.yaml")
    
    # Planning scene monitor configuration
    planning_scene_monitor_config = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    
    # Move group configuration
    move_group_config = {
        "planning_scene_monitor_options": planning_scene_monitor_config,
        "capabilities": "",
        "disable_capabilities": "",
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
        "default_planning_pipeline": "ompl",
        "planning_pipelines": ["ompl"],
    }
    
    # Move group node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics": kinematics_yaml},
            {"robot_description_planning": ompl_planning_yaml},
            joint_limits_yaml,
            moveit_controllers_yaml,
            move_group_config,
            {"use_sim_time": False},
        ],
    )
    
    return LaunchDescription(declared_arguments + [move_group_node])
