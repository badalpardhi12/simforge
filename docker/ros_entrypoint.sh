#!/bin/bash
# ROS 2 entrypoint script for Valid8 Cell

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

# Startup pipeline configuration
export ROBOT_IP=${ROBOT_IP:-192.168.1.9}
export UR_TYPE=${UR_TYPE:-ur5e}
export USE_MOCK_HARDWARE=${USE_MOCK_HARDWARE:-false}
export TF_PREFIX=${TF_PREFIX:-}

# Print startup info
echo "=============================================="
echo "Simforge Server Container - Valid8 Cell"
echo "=============================================="
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo "ROBOT_IP: $ROBOT_IP"
echo "UR_TYPE: $UR_TYPE"
echo "USE_MOCK_HARDWARE: $USE_MOCK_HARDWARE"
echo "REVERSE_IP: ${REVERSE_IP:-not set}"
echo "=============================================="

# Run startup pipeline if script exists and not skipped
STARTUP_SCRIPT="/ros2_ws/src/valid8_cell_control/scripts/startup_pipeline.py"
if [ -f "$STARTUP_SCRIPT" ] && [ "${SKIP_STARTUP_PIPELINE:-false}" != "true" ]; then
    echo ""
    echo "Running startup pipeline..."
    
    PIPELINE_ARGS="--robot-ip $ROBOT_IP --ur-type $UR_TYPE"
    
    if [ "$USE_MOCK_HARDWARE" = "true" ]; then
        PIPELINE_ARGS="$PIPELINE_ARGS --mock"
    fi
    
    if [ -n "$TF_PREFIX" ]; then
        PIPELINE_ARGS="$PIPELINE_ARGS --tf-prefix $TF_PREFIX"
    fi
    
    if [ -n "$REVERSE_IP" ]; then
        PIPELINE_ARGS="$PIPELINE_ARGS --reverse-ip $REVERSE_IP"
    fi
    
    python3 "$STARTUP_SCRIPT" $PIPELINE_ARGS || {
        echo "WARNING: Startup pipeline failed, continuing anyway..."
    }
    
    echo ""
fi

exec "$@"
