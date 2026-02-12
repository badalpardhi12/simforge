# SimForge Server — NVIDIA cuRobo Backend

GPU-accelerated motion planning for the Valid8 dual UR5e robot cell
using **NVIDIA cuRobo** for trajectory optimisation and collision-free
path planning.

## Features

- **cuRobo MotionGen** — GPU-accelerated trajectory optimisation (~30 ms planning)
- **cuRobo IKSolver** — batch inverse kinematics on GPU
- **Signed-distance-field collision checking** on GPU
- **Dual execution modes** — RTDE servoJ at 500 Hz (real) or
  `FollowJointTrajectory` via ros2_control (sim)
- **WebSocket protocol** on port 8766 (client-compatible with simforge_client)
- **Foxglove Bridge** on port 9090 for visualisation
- **Mode switching** (sim ↔ real) via signal files

## Directory Structure

```
simforge_server_nvidia/
├── CMakeLists.txt
├── package.xml
├── docker-compose.yml           # sim / prod profiles with GPU access
├── docker/
│   ├── Dockerfile               # CUDA 12.2 + ROS2 Humble + cuRobo + UR driver
│   └── ros_entrypoint.sh
├── launch/
│   ├── gateway.launch.py        # cuRobo gateway node
│   ├── sim.launch.py            # ros2_control with mock hardware
│   └── real.launch.py           # UR driver hardware interface
├── nodes/
│   └── command_gateway_curobo_node.py   # ROS2 node entry point
├── scripts/
│   └── start_server.sh          # Supervisor script
├── simforge_gateway/
│   ├── __init__.py
│   ├── config.py                # Robot configs (joints, IPs, limits)
│   ├── curobo_planner.py        # cuRobo MotionGen / IK wrapper
│   ├── protocol_executor.py     # Pose-by-pose fallback executor
│   ├── rpc_handlers.py          # WebSocket command dispatch
│   ├── rtde_controller.py       # ur_rtde servoJ / freedrive / IO
│   └── trajectory_executor.py   # Trajectory execution (sim + real)
├── config/                      # cuRobo YAML configs
│   ├── nakul_ur5e_curobo.yml
│   ├── sahadev_ur5e_curobo.yml
│   └── world_collision.yml
└── tests/
    └── test_curobo_gateway.py
```

## Quick Start

### Build & Run (Simulation)

```bash
cd simforge_server_nvidia
docker compose --profile sim up --build
```

### Build & Run (Real Robot)

```bash
NAKUL_ROBOT_IP=192.168.1.9 SAHADEV_ROBOT_IP=192.168.1.16 \
  docker compose --profile prod up --build
```

### Test

```bash
# Simulation only
python tests/test_curobo_gateway.py --sim-only

# Full sim → real test
python tests/test_curobo_gateway.py --server <SERVER_IP>
```

## GPU Requirements

- NVIDIA GPU with CUDA 12.2+ support
- NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- Tested on NVIDIA RTX Pro 6000 (x86_64)

## Configuration

### Velocity Scaling

```bash
MAX_VELOCITY_SCALING=0.15  # Slower (safer)
MAX_VELOCITY_SCALING=0.3   # Faster
```

### cuRobo Interpolation

```bash
INTERPOLATION_DT=0.02   # 50 Hz waypoints (default)
INTERPOLATION_DT=0.01   # 100 Hz waypoints (smoother)
```

## Architecture

```
┌──────────────┐     WebSocket :8766     ┌──────────────────────────┐
│ simforge_     │ ◄─────────────────────►│ command_gateway_curobo   │
│ client (GUI)  │                        │                          │
└──────────────┘                        │  rpc_handlers            │
                                        │       │                   │
┌──────────────┐     WebSocket :9090    │  curobo_planner (GPU)    │
│ Foxglove     │ ◄── foxglove_bridge    │       │                   │
│ Studio       │                        │  trajectory_executor     │
└──────────────┘                        │    ┌──────┴──────┐        │
                                        │    │             │        │
                                        │  (sim)        (real)     │
                                        └────┼─────────────┼────────┘
                                             │             │
                                     FollowJoint     ur_rtde servoJ
                                     Trajectory       (500 Hz RTDE)
                                             │
                                   ┌─────────▼──────────┐
                                   │ ros2_control_node   │
                                   │ (UR driver / mock)  │
                                   └────────────────────┘
```
