#!/bin/bash
set -e

# Auto-detect installed ROS2 distro (Humble on x86, Jazzy on Jetson)
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
elif [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
else
    echo "ERROR: No ROS2 distro found in /opt/ros/" >&2
    exit 1
fi

# Source the workspace if it's been built
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

exec "$@"
