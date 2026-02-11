#!/bin/bash
# ──────────────────────────────────────────────────────────────
# SimForge Server Startup Script — NVIDIA cuRobo Backend
#
# Starts the ROS2 stack (robots + controllers + Foxglove)
# in a managed subprocess, and then starts the cuRobo Command
# Gateway node independently.
#
# KEY DIFFERENCE from MoveIt version:
#   - MoveIt move_group is NOT launched or health-checked
#   - cuRobo initialisation happens INSIDE the gateway node
#   - Health check only requires joint_states + robot_description
#
# Usage (inside Docker):
#   /start_server.sh [--sim|--real]
# ──────────────────────────────────────────────────────────────
set -e

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

MODE="${1:---sim}"

NAKUL_IP="${NAKUL_ROBOT_IP:-192.168.1.9}"
SAHADEV_IP="${SAHADEV_ROBOT_IP:-192.168.1.16}"

# ── Signal files shared with the gateway ──
MODE_SWITCH_FILE="/tmp/simforge_mode_switch"
CURRENT_MODE_FILE="/tmp/simforge_current_mode"
STACK_READY_FILE="/tmp/simforge_stack_ready"

rm -f "$MODE_SWITCH_FILE" "$STACK_READY_FILE"

# ── Map flag to use_fake_hardware value ──
get_hardware_flag() {
    case "$1" in
        --sim|simulation) echo "true" ;;
        --real|real|both) echo "false" ;;
        *) echo "true" ;;
    esac
}

# ── Wait for ROS2 topics to be actively publishing ──
# NOTE: NO move_group check — cuRobo replaces MoveIt
wait_for_stack_health() {
    local max_wait=60
    local elapsed=0
    echo "Waiting for ROS2 stack to become healthy (cuRobo mode)..."

    while [ $elapsed -lt $max_wait ]; do
        # Check if /joint_states has publishers
        JS_PUBS=$(ros2 topic info /joint_states 2>/dev/null \
                  | grep "Publisher count:" | awk '{print $3}' || echo "0")

        # Check if /robot_description has publishers
        RD_PUBS=$(ros2 topic info /robot_description 2>/dev/null \
                  | grep "Publisher count:" | awk '{print $3}' || echo "0")

        echo "  [${elapsed}s] joint_states pubs=$JS_PUBS, robot_description pubs=$RD_PUBS"

        if [ "$JS_PUBS" -gt 0 ] 2>/dev/null && \
           [ "$RD_PUBS" -gt 0 ] 2>/dev/null; then
            echo "ROS2 stack is healthy! (no MoveIt — cuRobo runs inside gateway)"
            return 0
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done

    echo "WARNING: ROS2 stack health check timed out after ${max_wait}s"
    return 1
}

# ── Start the ROS2 stack (everything except gateway) ──
start_ros_stack() {
    local use_fake="$1"
    echo "============================================"
    echo "Starting ROS2 stack — cuRobo backend"
    echo "  use_fake_hardware=${use_fake}"
    echo "  Nakul IP:   $NAKUL_IP"
    echo "  Sahadev IP: $SAHADEV_IP"
    echo "============================================"

    rm -f "$STACK_READY_FILE"

    if [ "$use_fake" = "true" ]; then
        ros2 launch simforge_gateway_nvidia sim.launch.py \
            launch_foxglove:=true \
            launch_gateway:=false &
    else
        ros2 launch simforge_gateway_nvidia real.launch.py \
            nakul_robot_ip:="$NAKUL_IP" \
            sahadev_robot_ip:="$SAHADEV_IP" \
            launch_foxglove:=true \
            launch_gateway:=false \
            headless_mode:=true &
    fi

    ROS_STACK_PID=$!
    echo "ROS2 stack PID: $ROS_STACK_PID"

    if wait_for_stack_health; then
        echo "Stack health verified — signaling ready"
    else
        echo "Stack health check failed — signaling ready anyway (gateway will verify)"
    fi

    if [ "$use_fake" = "true" ]; then
        echo "simulation" > "$CURRENT_MODE_FILE"
    else
        echo "real" > "$CURRENT_MODE_FILE"
    fi
    echo "ready" > "$STACK_READY_FILE"
}

# ── Stop the ROS2 stack ──
stop_ros_stack() {
    rm -f "$STACK_READY_FILE"
    rm -f "$CURRENT_MODE_FILE"
    if [ -n "$ROS_STACK_PID" ] && kill -0 "$ROS_STACK_PID" 2>/dev/null; then
        echo "Stopping ROS2 stack (PID $ROS_STACK_PID)..."
        kill -INT "$ROS_STACK_PID" 2>/dev/null || true
        for i in $(seq 1 20); do
            if ! kill -0 "$ROS_STACK_PID" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "$ROS_STACK_PID" 2>/dev/null; then
            echo "Force killing ROS2 stack..."
            kill -9 "$ROS_STACK_PID" 2>/dev/null || true
        fi
        pkill -f "ros2_control_node" 2>/dev/null || true
        pkill -f "robot_state_publisher" 2>/dev/null || true
        pkill -f "spawner" 2>/dev/null || true
        pkill -f "foxglove_bridge" 2>/dev/null || true
        pkill -f "controller_stopper_node" 2>/dev/null || true
        pkill -f "urscript_interface" 2>/dev/null || true
        # NOTE: no pkill for move_group — cuRobo replaces it
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
    rm -f "$MODE_SWITCH_FILE" "$CURRENT_MODE_FILE" "$STACK_READY_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ── Start initial ROS2 stack ──
USE_FAKE=$(get_hardware_flag "$MODE")
start_ros_stack "$USE_FAKE"

echo "Waiting for ROS2 stack to initialise..."
sleep 10

# ── Start the cuRobo Command Gateway independently ──
echo "Starting cuRobo Command Gateway..."
ros2 run simforge_gateway_nvidia command_gateway_curobo_node.py \
    --ros-args \
    -p websocket_port:=8766 \
    -p websocket_host:=0.0.0.0 \
    -p max_clients:=5 \
    -p max_velocity_scaling:=${MAX_VELOCITY_SCALING:-0.25} \
    -p max_acceleration_scaling:=${MAX_ACCELERATION_SCALING:-0.25} \
    -p interpolation_dt:=${INTERPOLATION_DT:-0.02} &
GATEWAY_PID=$!
echo "Gateway PID: $GATEWAY_PID"

# ── Main loop: watch for mode switch requests ──
echo "Server running (cuRobo backend). Watching for mode switch requests..."
while true; do
    if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
        echo "Gateway process died, exiting..."
        exit 1
    fi

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
            echo "Mode switch complete."
        else
            echo "Already in requested mode, no switch needed"
        fi
    fi

    sleep 1
done
