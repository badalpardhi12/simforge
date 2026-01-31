"""
VLA Launch File

Launches VLA inference components.

Usage:
    ros2 launch simforge_server vla.launch.py
    ros2 launch simforge_server vla.launch.py model:=/models/openvla_int4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for VLA inference."""
    
    # Arguments
    model_path_arg = DeclareLaunchArgument(
        'model',
        default_value='/models/openvla',
        description='Path to VLA model'
    )
    
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='openvla/openvla-7b',
        description='Model name for HuggingFace'
    )
    
    use_tensorrt_arg = DeclareLaunchArgument(
        'use_tensorrt',
        default_value='false',
        description='Use TensorRT optimization'
    )
    
    precision_arg = DeclareLaunchArgument(
        'precision',
        default_value='fp16',
        description='Model precision (fp16, int8, int4)'
    )
    
    simulation_mode_arg = DeclareLaunchArgument(
        'simulation_mode',
        default_value='true',
        description='Run in simulation mode (no actual model)'
    )
    
    # VLA Inference node
    vla_node = Node(
        package='simforge_server',
        executable='vla_inference_node.py',
        name='vla_inference',
        output='screen',
        parameters=[{
            'model_name': LaunchConfiguration('model_name'),
            'model_path': LaunchConfiguration('model'),
            'use_tensorrt': LaunchConfiguration('use_tensorrt'),
            'precision': LaunchConfiguration('precision'),
            'simulation_mode': LaunchConfiguration('simulation_mode'),
        }],
    )
    
    return LaunchDescription([
        model_path_arg,
        model_name_arg,
        use_tensorrt_arg,
        precision_arg,
        simulation_mode_arg,
        vla_node,
    ])
