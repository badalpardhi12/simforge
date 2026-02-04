#!/usr/bin/env python3
"""
MoveIt 2 Planning Launch File for UR5e

This launch file starts the MoveIt 2 move_group node with:
- OMPL motion planning pipeline
- KDL/trac_ik kinematics solver
- Collision checking via planning scene
- Services for IK and motion planning

Services exposed:
- /compute_ik (moveit_msgs/srv/GetPositionIK)
- /plan_kinematic_path (moveit_msgs/srv/GetMotionPlan)
- /get_planning_scene (moveit_msgs/srv/GetPlanningScene)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for MoveIt 2 planning."""
    
    # Get package directories
    pkg_share = get_package_share_directory('simforge_server')
    
    # Paths to configuration files
    urdf_path = os.path.join(pkg_share, 'assets', 'ur5e', 'ur5e.urdf')
    srdf_path = os.path.join(pkg_share, 'config', 'moveit', 'ur5e.srdf')
    kinematics_yaml = os.path.join(pkg_share, 'config', 'moveit', 'kinematics.yaml')
    ompl_planning_yaml = os.path.join(pkg_share, 'config', 'moveit', 'ompl_planning.yaml')
    joint_limits_yaml = os.path.join(pkg_share, 'config', 'moveit', 'joint_limits.yaml')
    
    # Read URDF content
    with open(urdf_path, 'r') as urdf_file:
        robot_description_content = urdf_file.read()
    
    # Read SRDF content
    with open(srdf_path, 'r') as srdf_file:
        robot_description_semantic_content = srdf_file.read()
    
    # Robot description parameters
    robot_description = {'robot_description': robot_description_content}
    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_content}
    
    # MoveIt configuration parameters
    moveit_config = {
        'robot_description_planning': {
            'joint_limits': joint_limits_yaml,
        },
        'robot_description_kinematics': kinematics_yaml,
        'planning_scene_monitor_options': {
            'joint_state_topic': '/joint_states',
            'attached_collision_object_topic': '/attached_collision_objects',
            'publish_planning_scene_topic': '/planning_scene',
            'monitored_planning_scene_topic': '/planning_scene',
        },
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': '''default_planner_request_adapters/ResolveConstraintFrames
                default_planner_request_adapters/ValidateWorkspaceBounds
                default_planner_request_adapters/CheckStartStateBounds
                default_planner_request_adapters/CheckStartStateCollision''',
            'start_state_max_bounds_error': 0.1,
            'capabilities': '',
            'disable_capabilities': '',
            'default_planning_pipeline': 'ompl',
            'publish_robot_description': True,
            'publish_robot_description_semantic': True,
            'publish_geometry_updates': True,
            'publish_state_updates': True,
            'publish_transforms_updates': True,
            'monitor_dynamics': False,
        },
        'ompl': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
        },
    }
    
    # MoveIt move_group node
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            moveit_config,
            {'use_sim_time': True},
        ],
    )
    
    # Robot state publisher (for TF)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )
    
    return LaunchDescription([
        move_group_node,
        robot_state_publisher_node,
    ])
