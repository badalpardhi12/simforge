# SimForge Server — NVIDIA cuRobo Backend

GPU-accelerated motion planning for the Valid8 dual UR5e robot cell,
using **NVIDIA cuRobo** as a drop-in replacement for MoveIt2.

## What Changed

| Component | MoveIt2 Version | cuRobo Version |
|-----------|----------------|----------------|
| Motion planning | OMPL (sampling) | cuRobo MotionGen (trajectory optimisation on GPU) |
| IK solver | KDL / bio_ik | cuRobo IKSolver (batch IK on GPU) |
| Collision checking | FCL (CPU) | Signed distance fields (GPU) |
| Trajectory quality | Requires cosine-blended retiming (~260 lines) | Inherently smooth minimum-jerk trajectories |
| Planning time | ~500 ms typical | ~30 ms typical |
| Docker dependencies | `ros-humble-moveit`, OMPL, FCL | PyTorch, `nvidia-curobo` |
| `move_group` node | Required | **Not used** |

## What's Preserved

- ✅ WebSocket protocol on port 8766 (100% client-compatible)
- ✅ Foxglove Bridge on port 9090 (unchanged)
- ✅ ros2_control + UR driver (unchanged)
- ✅ TF2 transforms (unchanged)
- ✅ Mode switching (sim ↔ real) via signal files
- ✅ `FollowJointTrajectory` action for trajectory execution

## Directory Structure

```
simforge_server_nvidia/
├── CMakeLists.txt           # ROS2 ament_cmake package
├── package.xml
├── docker-compose.yml       # sim/prod profiles with GPU access
├── config/
│   ├── nakul_ur5e_curobo.yml   # cuRobo robot config (nakul)
│   ├── sahadev_ur5e_curobo.yml # cuRobo robot config (sahadev)
│   └── world_collision.yml     # Collision cuboids (table, floor, face)
├── docker/
│   ├── Dockerfile              # CUDA 12.2 + ROS2 Humble + cuRobo
│   └── ros_entrypoint.sh
├── launch/
│   ├── gateway.launch.py       # cuRobo gateway node
│   ├── sim.launch.py           # Simulation bringup (no MoveIt)
│   └── real.launch.py          # Real robot bringup (no MoveIt)
├── nodes/
│   └── command_gateway_curobo_node.py  # Main gateway (~900 lines vs ~2800)
├── scripts/
│   └── start_server.sh         # Supervisor (no move_group health check)
└── tests/
    └── test_curobo_gateway.py  # Automated sim→real test
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
- Tested on:
  - NVIDIA RTX Pro 6000 (x86_64)
  - NVIDIA Jetson Thor (aarch64 — use L4T base image variant)

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
┌──────────────┐     WebSocket :8766     ┌────────────────────────┐
│ simforge_     │ ◄──────────────────────►│ command_gateway_       │
│ client (GUI)  │                         │ curobo_node.py         │
└──────────────┘                         │                        │
                                         │  ┌─────────────────┐   │
┌──────────────┐     WebSocket :9090     │  │ cuRobo MotionGen│   │
│ Foxglove     │ ◄──── foxglove_bridge   │  │ (GPU / CUDA)    │   │
│ Studio       │                         │  └────────┬────────┘   │
└──────────────┘                         │           │             │
                                         │  FollowJointTrajectory │
                                         │           │             │
                                         └───────────┼─────────────┘
                                                     │
                                         ┌───────────▼─────────────┐
                                         │ ros2_control_node       │
                                         │ (UR driver / fake HW)   │
                                         └─────────────────────────┘
```
