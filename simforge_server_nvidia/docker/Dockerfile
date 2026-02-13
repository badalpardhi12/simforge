# ──────────────────────────────────────────────────────────────
# SimForge Server — NVIDIA cuRobo Backend
#
# GPU-accelerated motion planning with NVIDIA cuRobo.
# Real-robot control via ur_rtde (direct RTDE protocol).
# Simulated control via ros2_control (mock hardware).
#
# Build context: simforge_server_nvidia/
#
# Base: nvidia/cuda:12.8.0-devel-ubuntu22.04 + ROS2 Humble
# ──────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

LABEL maintainer="valid8.ai"
LABEL description="SimForge Server – NVIDIA cuRobo GPU Motion Planning"
LABEL version="2.0.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ── Locale ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8

# ── Essential tools ──────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl gnupg lsb-release wget git \
    cmake build-essential \
    python3-pip python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ── Install ROS2 Humble ─────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-std-srvs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-trajectory-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-control-msgs \
    ros-humble-visualization-msgs \
    ros-humble-nav-msgs \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-urdf \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# ── UR Robot Driver + ros2_control ───────────────────────────
# Required for:
#   - ros2_control_node (fake hardware in sim, UR driver in real)
#   - joint_state_broadcaster (/joint_states in sim mode)
#   - scaled_joint_trajectory_controller (FollowJointTrajectory in sim)
#   - controller_stopper_node (headless mode URScript re-send)
#   - ur_description URDF macros
RUN apt-get update && apt-get install -y \
    ros-humble-ur-robot-driver \
    ros-humble-ur-description \
    ros-humble-ur-calibration \
    ros-humble-controller-manager \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-force-torque-sensor-broadcaster \
    && rm -rf /var/lib/apt/lists/*

# ── Foxglove Bridge ─────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# ── Additional ROS2 deps ────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ros-humble-kdl-parser \
    ros-humble-ament-cmake-python \
    && rm -rf /var/lib/apt/lists/*

# Initialise rosdep
RUN rosdep init || true && rosdep update

# ── PyTorch ──────────────────────────────────────────────────
RUN pip3 install --no-cache-dir torch

# ── NVIDIA cuRobo (from source) ─────────────────────────────
RUN apt-get update && apt-get install -y git-lfs ninja-build && rm -rf /var/lib/apt/lists/*
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
RUN git lfs install && \
    git clone https://github.com/NVlabs/curobo.git /tmp/curobo && \
    cd /tmp/curobo && \
    pip3 install -e . --no-build-isolation && \
    rm -rf /tmp/curobo/.git

# ── Python dependencies for gateway ─────────────────────────
RUN apt-get update && apt-get install -y \
    libboost-system-dev libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    websockets>=12.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    transforms3d>=0.4.1 \
    pyyaml>=6.0 \
    ur-rtde>=1.5.0

# ── Create workspace ────────────────────────────────────────
WORKDIR /ros2_ws

# ── Source layers (ordered least→most frequently changed) ────

# Layer 1 — Cell description / control packages (rarely change)
COPY valid8_dual_cell_description /ros2_ws/src/valid8_dual_cell_description
COPY valid8_dual_cell_control     /ros2_ws/src/valid8_dual_cell_control
COPY face_robot_ur20_description  /ros2_ws/src/face_robot_ur20_description
COPY face_robot_ur20_control      /ros2_ws/src/face_robot_ur20_control

# Layer 2 — Gateway package manifest (changes when deps change)
COPY CMakeLists.txt  /ros2_ws/src/simforge_gateway_nvidia/CMakeLists.txt
COPY package.xml     /ros2_ws/src/simforge_gateway_nvidia/package.xml

# Layer 3 — Config + launch (change occasionally)
COPY config /ros2_ws/src/simforge_gateway_nvidia/config
COPY launch /ros2_ws/src/simforge_gateway_nvidia/launch

# Layer 4 — Python package + nodes + tests (change most often)
COPY simforge_gateway_nvidia /ros2_ws/src/simforge_gateway_nvidia/simforge_gateway_nvidia
COPY nodes  /ros2_ws/src/simforge_gateway_nvidia/nodes
COPY tests  /ros2_ws/src/simforge_gateway_nvidia/tests

# ── Build ROS2 packages ─────────────────────────────────────
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /ros2_ws && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# ── Entrypoint & start script ───────────────────────────────
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

COPY scripts/start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

# ── Environment ──────────────────────────────────────────────
ENV ROS_DOMAIN_ID=42
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 9090 — Foxglove Bridge WebSocket
# 8766 — cuRobo Command Gateway WebSocket
EXPOSE 9090
EXPOSE 8766

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
