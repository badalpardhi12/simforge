"""
MoveIt2 Move Group Launch File for Valid8 Dual Cell

Launches the MoveIt2 move_group node for motion planning with both robots.

YAML config files are loaded as Python dicts and passed directly as
parameters so that move_group receives them correctly (not as --params-file
which requires the ``node_name: ros__parameters:`` wrapper).
"""
import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name: str, file_path: str) -> dict:
    """Load a YAML file from a ROS2 package share directory."""
    pkg_dir = get_package_share_directory(package_name)
    abs_path = os.path.join(pkg_dir, file_path)
    with open(abs_path, "r") as f:
        return yaml.safe_load(f)


def launch_setup(context, *args, **kwargs):
    use_sim = LaunchConfiguration("use_sim")
    launch_rviz = LaunchConfiguration("launch_rviz")

    nakul_ur_type = LaunchConfiguration("nakul_ur_type")
    sahadev_ur_type = LaunchConfiguration("sahadev_ur_type")
    nakul_robot_ip = LaunchConfiguration("nakul_robot_ip")
    sahadev_robot_ip = LaunchConfiguration("sahadev_robot_ip")

    nakul_kinematics_file = LaunchConfiguration("nakul_kinematics_parameters_file")
    sahadev_kinematics_file = LaunchConfiguration(
        "sahadev_kinematics_parameters_file"
    )

    # ── Robot description (URDF via xacro) ───────────────────────────
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
            "nakul_ur_type:=", nakul_ur_type,
            " ",
            "sahadev_ur_type:=", sahadev_ur_type,
            " ",
            "nakul_use_fake_hardware:=", use_sim,
            " ",
            "sahadev_use_fake_hardware:=", use_sim,
            " ",
            "nakul_kinematics_parameters_file:=", nakul_kinematics_file,
            " ",
            "sahadev_kinematics_parameters_file:=", sahadev_kinematics_file,
            " ",
            "nakul_fake_sensor_commands:=false",
            " ",
            "sahadev_fake_sensor_commands:=false",
            " ",
            "headless_mode:=false",
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(
            robot_description_content, value_type=str
        )
    }

    # ── SRDF ─────────────────────────────────────────────────────────
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("valid8_dual_cell_moveit_config"),
                    "srdf",
                    "valid8_dual_cell.srdf",
                ]
            ),
        ]
    )
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(
            robot_description_semantic_content, value_type=str
        )
    }

    # ── Load YAML configs as Python dicts ────────────────────────────
    kinematics_yaml = load_yaml(
        "valid8_dual_cell_moveit_config", "config/kinematics.yaml"
    )
    ompl_planning_yaml = load_yaml(
        "valid8_dual_cell_moveit_config", "config/ompl_planning.yaml"
    )
    joint_limits_yaml = load_yaml(
        "valid8_dual_cell_moveit_config", "config/joint_limits.yaml"
    )
    moveit_controllers_yaml = load_yaml(
        "valid8_dual_cell_moveit_config", "config/moveit_controllers.yaml"
    )

    # ── Planning pipeline configuration ──────────────────────────────
    # MoveIt2 Humble loads planning pipeline params from the namespace
    # matching the pipeline name.  default_planning_pipeline is "ompl",
    # so MoveIt looks for "ompl.planning_plugin", "ompl.request_adapters", etc.
    planning_pipelines = {
        "default_planning_pipeline": "ompl",
        "planning_pipelines": ["ompl"],
    }

    # Put OMPL config under the "ompl" namespace
    ompl_pipeline_config = {
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": "default_planner_request_adapters/FixWorkspaceBounds "
                                "default_planner_request_adapters/FixStartStateBounds "
                                "default_planner_request_adapters/FixStartStateCollision "
                                "default_planner_request_adapters/FixStartStatePathConstraints "
                                "default_planner_request_adapters/ResolveConstraintFrames "
                                "default_planner_request_adapters/AddTimeOptimalParameterization",
            "start_state_max_bounds_error": 0.1,
            # ── TOTG (Time-Optimal Trajectory Generation) tuning ─────
            # resample_dt=0.1 produces ~10 waypoints/sec with 100ms
            # spacing.  This is optimal for the ScaledJointTrajectory
            # Controller's spline interpolation because:
            #   - Wider spacing reduces cubic spline coefficient
            #     magnification (c2 ~ 1/T^2, c3 ~ 1/T^3)
            #   - Fewer segment boundaries = fewer acceleration
            #     discontinuities = less jerkiness
            #   - 0.02 (50 pts/sec) was COUNTERPRODUCTIVE — 50
            #     acceleration discontinuities per second at segment
            #     boundaries, with spline coefficients amplified by
            #     1/0.02^3 = 125,000
            #
            # path_tolerance=0.1 allows TOTG to round corners at
            # intermediate waypoints for C1 continuity.
            #
            # NOTE: This MoveIt2 Humble build uses FLAT param names
            # (ompl.resample_dt), NOT the nested totg. namespace.
            "path_tolerance": 0.1,
            "resample_dt": 0.1,
            "min_angle_change": 0.001,
        }
    }

    # Merge planner_configs and group configs from ompl_planning.yaml into ompl namespace
    if "planner_configs" in ompl_planning_yaml:
        ompl_pipeline_config["ompl"]["planner_configs"] = ompl_planning_yaml["planner_configs"]
    # Copy group-specific configs (nakul_arm, sahadev_arm, dual_arms)
    for key in ompl_planning_yaml:
        if key not in ("planning_plugin", "request_adapters", "start_state_max_bounds_error", "planner_configs"):
            ompl_pipeline_config["ompl"][key] = ompl_planning_yaml[key]

    # ── Trajectory execution settings ────────────────────────────────
    # Must match old server:
    #   - moveit_manage_controllers: False (UR driver manages controllers)
    #   - allowed_start_tolerance: 0.1 rad (0.01 is too strict)
    #   - execution_duration_monitoring: False (scaled JTC timing mismatch)
    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.1,
        "trajectory_execution.execution_duration_monitoring": False,
    }

    # ── Planning scene monitor settings ──────────────────────────────
    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
    }

    # ── CRITICAL: Parameter wrapping ─────────────────────────────────
    # MoveIt expects kinematics under "robot_description_kinematics"
    # and joint limits under "robot_description_planning".  Without
    # these wrappers MoveIt cannot find the IK solver or joint limits.
    robot_description_kinematics = {
        "robot_description_kinematics": kinematics_yaml,
    }
    robot_description_planning = {
        "robot_description_planning": joint_limits_yaml,
    }

    # ── Old-style move_group namespace config (backwards compat) ─────
    # Some MoveIt Humble builds still look under "move_group" namespace.
    ompl_old_style_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters":
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints "
                "default_planner_request_adapters/ResolveConstraintFrames "
                "default_planner_request_adapters/AddTimeOptimalParameterization",
            "start_state_max_bounds_error": 0.1,
            # TOTG params — match ompl_pipeline_config
            "path_tolerance": 0.1,
            "resample_dt": 0.1,
            "min_angle_change": 0.001,
        }
    }
    # Merge planner configs into old-style too
    if ompl_planning_yaml:
        for k, v in ompl_planning_yaml.items():
            ompl_old_style_config["move_group"][k] = v

    # ── Move group node ──────────────────────────────────────────────
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            ompl_pipeline_config,
            planning_pipelines,
            ompl_old_style_config,
            moveit_controllers_yaml,
            trajectory_execution,
            planning_scene_monitor_parameters,
            {"use_sim_time": use_sim},
        ],
    )

    # ── RViz (optional) ──────────────────────────────────────────────
    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare("valid8_dual_cell_moveit_config"),
            "config",
            "moveit.rviz",
        ]
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"use_sim_time": use_sim},
        ],
        condition=IfCondition(launch_rviz),
    )

    return [move_group_node, rviz_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "use_sim",
            default_value="false",
            description="Use simulation mode with mock hardware.",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz with MoveIt configuration.",
        ),
        DeclareLaunchArgument(
            "nakul_ur_type",
            default_value="ur5e",
        ),
        DeclareLaunchArgument(
            "sahadev_ur_type",
            default_value="ur5e",
        ),
        DeclareLaunchArgument(
            "nakul_robot_ip",
            default_value="192.168.1.9",
        ),
        DeclareLaunchArgument(
            "sahadev_robot_ip",
            default_value="192.168.1.16",
        ),
        DeclareLaunchArgument(
            "nakul_kinematics_parameters_file",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("valid8_dual_cell_control"),
                    "config",
                    "nakul_calibration.yaml",
                ]
            ),
        ),
        DeclareLaunchArgument(
            "sahadev_kinematics_parameters_file",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("valid8_dual_cell_control"),
                    "config",
                    "sahadev_calibration.yaml",
                ]
            ),
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
