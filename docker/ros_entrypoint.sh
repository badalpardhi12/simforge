#!/bin/bash
# ROS 2 entrypoint script

set -e

# Source ROS 2 setup
source /opt/ros/humble/setup.bash

# Source workspace setup if it exists
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Setup environment variables
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}

# Print startup info
echo "=============================================="
echo "Simforge Server Container"
echo "=============================================="
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo "=============================================="

exec "$@"
