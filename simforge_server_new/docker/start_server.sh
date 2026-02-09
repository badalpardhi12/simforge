#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Simforge Server Startup Script
#
# Starts the ROS2 stack (robots + controllers + MoveIt + Foxglove)
# in a managed subprocess, and then starts the Command Gateway
# node independently.
#
# The gateway can signal a mode switch by writing to
#   /tmp/simforge_mode_switch
# This script will then kill the ROS2 stack and restart it with
# the new hardware mode.
#
# Usage (inside Docker):
#   /start_server.sh [--sim|--real]
# ──────────────────────────────────────────────────────────────
set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

# Default to simulation mode
MODE="${1:---sim}"

NAKUL_IP="${NAKUL_ROBOT_IP:-192.168.1.9}"
SAHADEV_IP="${SAHADEV_ROBOT_IP:-192.168.1.16}"

# ── Signal file used by the gateway to request mode switch ──
MODE_SWITCH_FILE="/tmp/simforge_mode_switch"
CURRENT_MODE_FILE="/tmp/simforge_current_mode"

rm -f "$MODE_SWITCH_FILE"

# ── Map flag to use_fake_hardware value ──
get_hardware_flag() {
    case "$1" in
        --sim|simulation) echo "true" ;;
        --real|real|both) echo "false" ;;
        *) echo "true" ;;
    esac
}

# ── Start the ROS2 stack (everything except gateway) ──
start_ros_stack() {
    local use_fake="$1"
    echo "============================================"
    echo "Starting ROS2 stack (use_fake_hardware=${use_fake})"
    echo "  Nakul IP:   $NAKUL_IP"
    echo "  Sahadev IP: $SAHADEV_IP"
    echo "============================================"

    if [ "$use_fake" = "true" ]; then
        # Simulation mode — use sim.launch.py without the gateway
        ros2 launch valid8_dual_cell_bringup sim.launch.py \
            launch_rviz:=false \
            launch_foxglove:=true \
            launch_gateway:=false \
            launch_moveit:=true &
    else
        # Real robot mode — use real.launch.py without the gateway
        ros2 launch valid8_dual_cell_bringup real.launch.py \
            nakul_robot_ip:="$NAKUL_IP" \
            sahadev_robot_ip:="$SAHADEV_IP" \
            launch_rviz:=false \
            launch_foxglove:=true \
            launch_gateway:=false \
            launch_moveit:=true \
            headless_mode:=true &
    fi

    ROS_STACK_PID=$!
    echo "ROS2 stack PID: $ROS_STACK_PID"

    # Record current mode
    if [ "$use_fake" = "true" ]; then
        echo "simulation" > "$CURRENT_MODE_FILE"
    else
        echo "real" > "$CURRENT_MODE_FILE"
    fi
}

# ── Stop the ROS2 stack ──
stop_ros_stack() {
    if [ -n "$ROS_STACK_PID" ] && kill -0 "$ROS_STACK_PID" 2>/dev/null; then
        echo "Stopping ROS2 stack (PID $ROS_STACK_PID)..."
        # Send SIGINT first for graceful shutdown
        kill -INT "$ROS_STACK_PID" 2>/dev/null || true
        # Wait up to 10 seconds for graceful shutdown
        for i in $(seq 1 20); do
            if ! kill -0 "$ROS_STACK_PID" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        # Force kill if still running
        if kill -0 "$ROS_STACK_PID" 2>/dev/null; then
            echo "Force killing ROS2 stack..."
            kill -9 "$ROS_STACK_PID" 2>/dev/null || true
        fi
        # Also kill any lingering ros2/move_group/controller_manager processes
        pkill -f "ros2_control_node" 2>/dev/null || true
        pkill -f "move_group" 2>/dev/null || true
        pkill -f "robot_state_publisher" 2>/dev/null || true
        pkill -f "spawner" 2>/dev/null || true
        pkill -f "foxglove_bridge" 2>/dev/null || true
        pkill -f "controller_stopper_node" 2>/dev/null || true
        pkill -f "urscript_interface" 2>/dev/null || true
        # Wait for ports (50001-50008, 9090) to be released
        sleep 3
        echo "ROS2 stack stopped"
    fi
}

# ── Cleanup on exit ──
cleanup() {
    echo "Shutting down..."
    stop_ros_stack
    if [ -n "$GATEWAY_PID" ] && kill -0 "$GATEWAY_PID" 2>/dev/null; then
        kill "$GATEWAY_PID" 2>/dev/null || true
    fi
    rm -f "$MODE_SWITCH_FILE" "$CURRENT_MODE_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ── Start initial ROS2 stack ──
USE_FAKE=$(get_hardware_flag "$MODE")
start_ros_stack "$USE_FAKE"

# Wait for the stack to initialize before starting gateway
echo "Waiting for ROS2 stack to initialize..."
sleep 10

# ── Start the Command Gateway independently ──
echo "Starting Command Gateway..."
ros2 run simforge_gateway command_gateway_node.py \
    --ros-args \
    -p websocket_port:=8766 \
    -p websocket_host:=0.0.0.0 \
    -p max_clients:=5 &
GATEWAY_PID=$!
echo "Gateway PID: $GATEWAY_PID"

# ── Main loop: watch for mode switch requests ──
echo "Server running. Watching for mode switch requests..."
while true; do
    # Check if gateway is still alive
    if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
        echo "Gateway process died, exiting..."
        exit 1
    fi

    # Check for mode switch request
    if [ -f "$MODE_SWITCH_FILE" ]; then
        NEW_MODE=$(cat "$MODE_SWITCH_FILE")
        rm -f "$MODE_SWITCH_FILE"
        echo "Mode switch requested: $NEW_MODE"

        NEW_FAKE=$(get_hardware_flag "$NEW_MODE")
        CURRENT_FAKE=$(get_hardware_flag "$(cat $CURRENT_MODE_FILE 2>/dev/null || echo simulation)")

        if [ "$NEW_FAKE" != "$CURRENT_FAKE" ]; then
            echo "Switching hardware mode..."
            stop_ros_stack
            start_ros_stack "$NEW_FAKE"
            echo "Mode switch complete. Waiting for services..."
        else
            echo "Already in requested mode, no switch needed"
        fi
    fi

    sleep 1
done
