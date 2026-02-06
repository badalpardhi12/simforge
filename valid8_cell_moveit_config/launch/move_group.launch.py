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
    """Load a yaml file from a package.
    
    Tries multiple paths to find config files, since Docker volume mounts
    can break the ament index but the files are still accessible at known paths.
    """
    # Try multiple paths in order of preference:
    paths_to_try = []
    
    # 1. Try ament index (standard ROS2 way)
    try:
        from ament_index_python.packages import get_package_share_directory
        package_dir = get_package_share_directory(package_name)
        paths_to_try.append(os.path.join(package_dir, file_path))
    except Exception:
        pass
    
    # 2. Try installed share directory directly (Docker volume mount may override ament index)
    paths_to_try.append(f"/ros2_ws/install/{package_name}/share/{package_name}/{file_path}")
    
    # 3. Try source directory (for development)
    paths_to_try.append(f"/ros2_ws/src/{package_name}/{file_path}")
    
    for full_path in paths_to_try:
        try:
            with open(full_path, 'r') as file:
                data = yaml.safe_load(file)
                if data:
                    print(f"Loaded config: {full_path}")
                    return data
        except Exception:
            continue
    
    print(f"WARNING: Could not load {package_name}/{file_path} from any path!")
    print(f"  Tried: {paths_to_try}")
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
    # Try multiple paths for SRDF since ament index may not work with Docker volume mounts
    srdf_content = None
    srdf_paths = [
        "/ros2_ws/install/valid8_cell_moveit_config/share/valid8_cell_moveit_config/config/valid8_cell.srdf",
        "/ros2_ws/src/valid8_cell_moveit_config/config/valid8_cell.srdf",
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        moveit_config_dir = get_package_share_directory("valid8_cell_moveit_config")
        srdf_paths.insert(0, os.path.join(moveit_config_dir, "config", "valid8_cell.srdf"))
    except Exception:
        pass
    
    for srdf_path in srdf_paths:
        try:
            with open(srdf_path, 'r') as f:
                srdf_content = f.read()
                print(f"Loaded SRDF: {srdf_path}")
                break
        except Exception:
            continue
    
    if not srdf_content:
        raise RuntimeError(f"Could not find SRDF file! Tried: {srdf_paths}")
    
    robot_description_semantic = {"robot_description_semantic": srdf_content}
    
    # Load other MoveIt configs
    kinematics_yaml = load_yaml("valid8_cell_moveit_config", "config/kinematics.yaml")
    ompl_planning_yaml = load_yaml("valid8_cell_moveit_config", "config/ompl_planning.yaml")
    joint_limits_yaml = load_yaml("valid8_cell_moveit_config", "config/joint_limits.yaml")
    moveit_controllers_yaml = load_yaml("valid8_cell_moveit_config", "config/moveit_controllers.yaml")
    
    # CRITICAL: Joint limits must be wrapped under "robot_description_planning" namespace.
    # MoveIt's C++ code reads default_velocity_scaling_factor, default_acceleration_scaling_factor,
    # and per-joint limits from this namespace. Without it, MoveIt ignores our overrides
    # and falls back to raw URDF limits (which have no acceleration limits or scaling).
    # This matches what MoveItConfigsBuilder.joint_limits() does internally.
    robot_description_planning = {
        "robot_description_planning": joint_limits_yaml,
    }
    
    # OMPL Planning Pipeline Configuration
    # MoveIt Humble expects planning pipeline config in a specific format:
    # - "planning_pipelines": list of pipeline names
    # - "default_planning_pipeline": which one to use by default
    # - "<pipeline_name>": dict with planning_plugin, request_adapters, and planner configs
    # This matches what MoveItConfigsBuilder.planning_pipelines() produces.
    ompl_config = {
        "planning_plugin": "ompl_interface/OMPLPlanner",
        "request_adapters": "default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints",
        "start_state_max_bounds_error": 0.1,
    }
    # Merge OMPL planner-specific configs (planner_configs, etc.)
    if ompl_planning_yaml:
        ompl_config.update(ompl_planning_yaml)
    
    planning_pipelines_config = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl_config,
    }
    
    # Also keep the old-style "move_group" namespace config for backwards compatibility
    # (some MoveIt Humble builds still look here)
    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": "default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints",
            "start_state_max_bounds_error": 0.1,
        }
    }
    if ompl_planning_yaml:
        ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)
    
    # Move group configuration
    move_group_config = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
    }
    
    # Trajectory execution configuration
    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.1,
        # Execution time monitoring can be incompatible with the scaled JTC
        "trajectory_execution.execution_duration_monitoring": False,
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
            robot_description_planning,
            planning_pipelines_config,
            ompl_planning_pipeline_config,
            moveit_controllers_yaml,
            move_group_config,
            trajectory_execution,
            {"use_sim_time": False},
        ],
    )
    
    return LaunchDescription(declared_arguments + [move_group_node])
