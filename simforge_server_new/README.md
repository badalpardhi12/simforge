# Simforge Server New - Dual UR5e Robot Cell

This is a self-contained ROS2 Humble server for controlling dual UR5e robots with MoveIt2, Foxglove Bridge visualization, and support for both simulation and real hardware.

## Architecture

```
simforge_server_new/
├── docker/                    # Docker configuration
│   ├── Dockerfile            # Main server Dockerfile
│   └── docker-compose.yml    # Compose configuration
├── valid8_dual_cell_description/   # URDF/XACRO robot description
├── valid8_dual_cell_control/       # ros2_control configuration
├── valid8_dual_cell_moveit_config/ # MoveIt2 configuration
├── simforge_gateway/              # WebSocket gateway for client
├── meshes/                        # 3D mesh files
└── config/                        # Shared configuration files
```

## Robots

- **nakul_ur5e**: IP 192.168.1.9, Position [-0.6758, 0, 1.03] with -90° yaw
- **sahadev_ur5e**: IP 192.168.1.16, Position [0.6758, 0, 1.03] with +90° yaw

## Quick Start

### Build Docker Image
```bash
cd simforge_server_new
docker compose build
```

### Run in Simulation Mode
```bash
docker compose up simforge-server-sim
```

### Run with Real Robots
```bash
docker compose up simforge-server
```

### Connect from macOS Client
The server exposes:
- Port 9090: Foxglove Bridge (visualization)
- Port 8765: Command Gateway WebSocket

## Launch Files

### Simulation Mode
```bash
ros2 launch valid8_dual_cell_control start_robots.launch.py \
    nakul_use_mock_hardware:=true \
    sahadev_use_mock_hardware:=true
```

### Real Hardware
```bash
ros2 launch valid8_dual_cell_control start_robots.launch.py \
    nakul_robot_ip:=192.168.1.9 \
    sahadev_robot_ip:=192.168.1.16
```

### With MoveIt2
```bash
ros2 launch valid8_dual_cell_moveit_config move_group.launch.py
```

## Foxglove Visualization

Connect Foxglove Studio to `ws://<server-ip>:9090` to visualize:
- Robot state and joint positions
- TF transforms
- Camera feeds (if available)
- Planned trajectories

## Based On

- [Universal Robots ROS2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)
- [UR ROS2 Tutorials - Dual Robot Cell](https://github.com/UniversalRobots/Universal_Robots_ROS2_Tutorials/tree/main/my_dual_robot_cell)
- [MoveIt2](https://moveit.ros.org/)
