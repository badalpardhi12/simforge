#!/bin/bash
# ROS2 Entrypoint Script for Simforge Server

set -e

# Setup ROS2 environment
source /opt/ros/humble/setup.bash

# Source workspace if built
if [ -f "/ros2_ws/install/setup.bash" ]; then
    source /ros2_ws/install/setup.bash
fi

# Execute the command
exec "$@"
