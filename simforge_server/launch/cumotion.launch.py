#!/usr/bin/env python3
"""
cuMotion Launch Configuration for Simforge

Launches Isaac ROS cuMotion for GPU-accelerated motion planning.
Integrates with MoveIt 2 as a planning plugin and uses nvblox ESDF
for real-time collision avoidance.

This launch file supports:
- UR5e robot with cuMotion planning
- nvblox ESDF integration
- MoveIt 2 planning interface
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for cuMotion with UR5e."""
    
    # Launch arguments
    robot_name = LaunchConfiguration('robot_name', default='nakul_ur5e')
    use_nvblox = LaunchConfiguration('use_nvblox', default='true')
    
    # Config file paths
    config_dir = os.path.join(
        get_package_share_directory('simforge_server'),
        'config', 'cumotion'
    )
    
    # Declare launch arguments
    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='nakul_ur5e',
        description='Name of the robot'
    )
    
    declare_use_nvblox = DeclareLaunchArgument(
        'use_nvblox',
        default_value='true',
        description='Enable nvblox ESDF for collision checking'
    )
    
    # Load cuMotion configuration
    cumotion_config = os.path.join(config_dir, 'ur5e_cumotion.yaml')
    
    # cuMotion planner node
    cumotion_planner_node = Node(
        package='isaac_ros_cumotion',
        executable='cumotion_planner_node',
        name='cumotion_planner',
        output='screen',
        parameters=[
            cumotion_config,
            {
                'robot_name': robot_name,
                'use_nvblox_costmap': use_nvblox,
            }
        ],
        remappings=[
            # Connect to nvblox ESDF
            ('esdf_costmap', '/nvblox/combined_esdf'),
            # Robot state
            ('joint_states', '/joint_states'),
            # Planning services
            ('plan_kinematic_path', '/cumotion/plan_kinematic_path'),
        ],
    )
    
    # MoveIt 2 integration node (provides MoveGroup interface)
    moveit_cumotion_node = Node(
        package='isaac_ros_cumotion_moveit',
        executable='cumotion_moveit_plugin',
        name='cumotion_moveit',
        output='screen',
        parameters=[
            cumotion_config,
            {
                'robot_description': os.path.join(
                    get_package_share_directory('simforge_server'),
                    'assets', 'ur5e', 'ur5e.urdf'
                ),
            }
        ],
    )
    
    # Motion planner service node (custom wrapper)
    motion_planner_node = Node(
        package='simforge_server',
        executable='motion_planner_node',
        name='motion_planner',
        output='screen',
        parameters=[{
            'robot_name': robot_name,
            'base_link': 'base_link',
            'tip_link': 'tool0',
            'use_nvblox': use_nvblox,
            'nvblox_costmap_topic': '/nvblox/combined_esdf',
        }],
    )
    
    return LaunchDescription([
        declare_robot_name,
        declare_use_nvblox,
        cumotion_planner_node,
        moveit_cumotion_node,
        motion_planner_node,
    ])
