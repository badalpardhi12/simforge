# Simforge Server New - Dual UR5e Robot Cell
#
# Docker image with ROS2 Humble, MoveIt2, UR Robot Driver, and Foxglove Bridge
# Based on Universal Robots ROS2 Tutorials

FROM ros:humble-ros-base-jammy

LABEL maintainer="valid8.ai"
LABEL description="Simforge Server - Dual UR5e Robot Cell with ROS2 Humble"
LABEL version="2.0.0"

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8

# ============================================================
# Install essential tools
# ============================================================
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg \
    lsb-release \
    wget \
    git \
    cmake \
    build-essential \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-colcon-common-extensions \
    python3-rosdep \
    libboost-all-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install ROS2 Humble packages
# ============================================================
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-rclpy \
    ros-humble-std-msgs \
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
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-urdf \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install MoveIt2
# ============================================================
RUN apt-get update && apt-get install -y \
    ros-humble-moveit \
    ros-humble-moveit-planners \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-servo \
    ros-humble-moveit-setup-assistant \
    ros-humble-moveit-configs-utils \
    ros-humble-moveit-simple-controller-manager \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install Universal Robots ROS2 Driver and dependencies
# ============================================================
RUN apt-get update && apt-get install -y \
    ros-humble-ur-robot-driver \
    ros-humble-ur-description \
    ros-humble-ur-moveit-config \
    ros-humble-ur-calibration \
    ros-humble-controller-manager \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-force-torque-sensor-broadcaster \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install Foxglove Bridge for visualization
# ============================================================
RUN apt-get update && apt-get install -y \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install additional dependencies
# ============================================================
RUN apt-get update && apt-get install -y \
    ros-humble-kdl-parser \
    ros-humble-ament-cmake-python \
    libnlopt-dev \
    libnlopt-cxx-dev \
    liborocos-kdl-dev \
    python3-pykdl \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true && rosdep update

# ============================================================
# Install Python dependencies for gateway
# ============================================================
RUN pip3 install --no-cache-dir \
    websockets>=12.0 \
    asyncio \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    transforms3d>=0.4.1 \
    pyyaml>=6.0 \
    foxglove-websocket>=0.1.0

# ============================================================
# Create workspace directory
# ============================================================
WORKDIR /ros2_ws

# ============================================================
# Copy package sources
# ============================================================

# Copy valid8 dual cell packages
COPY valid8_dual_cell_description /ros2_ws/src/valid8_dual_cell_description
COPY valid8_dual_cell_control /ros2_ws/src/valid8_dual_cell_control
COPY valid8_dual_cell_moveit_config /ros2_ws/src/valid8_dual_cell_moveit_config
COPY valid8_dual_cell_bringup /ros2_ws/src/valid8_dual_cell_bringup

# Copy gateway node
COPY simforge_gateway /ros2_ws/src/simforge_gateway

# ============================================================
# Build ROS2 packages
# ============================================================
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /ros2_ws && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# ============================================================
# Setup entrypoint
# ============================================================
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

COPY docker/start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

# ============================================================
# Environment variables
# ============================================================
ENV ROS_DOMAIN_ID=42
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# ============================================================
# Expose ports
# ============================================================
# Foxglove Bridge WebSocket
EXPOSE 9090
# Command Gateway WebSocket
EXPOSE 8765
# ROS2 DDS ports
EXPOSE 7400-7500/udp

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
