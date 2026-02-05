"""
MoveIt Setup Assistant Launch File

Use this to regenerate MoveIt configuration using the graphical Setup Assistant.
This will allow you to:
- Generate/update SRDF with proper collision matrix
- Configure planning groups
- Add named poses
- Configure end effectors
- Set up controllers

Usage:
    ros2 launch valid8_cell_moveit_config setup_assistant.launch.py

Reference: https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html
"""

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_setup_assistant_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "valid8_cell", package_name="valid8_cell_moveit_config"
    ).to_moveit_configs()
    return generate_setup_assistant_launch(moveit_config)
