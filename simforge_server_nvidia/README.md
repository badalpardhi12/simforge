# SimForge Server — NVIDIA cuRobo Backend

GPU-accelerated motion planning for robotic workcells using **NVIDIA cuRobo**
for trajectory optimisation and collision-free path planning, with real-robot
control via **ur_rtde** (500 Hz servoJ) and simulated control via
**ros2_control** (mock hardware).

## Features

- **cuRobo MotionGen** — GPU-accelerated trajectory optimisation (~30 ms planning)
- **cuRobo IKSolver** — batch inverse kinematics on GPU
- **Signed-distance-field collision checking** on GPU
- **Multi-environment support** — switch between robot configurations via
  `ENV_CONFIG` environment variable
- **Dual execution modes** — RTDE servoJ at 500 Hz (real) or
  `FollowJointTrajectory` via ros2_control (sim)
- **WebSocket protocol** on port 8766 (client-compatible with simforge_client)
- **Foxglove Bridge** on port 9090 for visualisation
- **Mode switching** (sim ↔ real) via signal files

## Supported Environments

| Environment        | Description                           | Robots              |
|--------------------|---------------------------------------|----------------------|
| `valid8_dual_ur5e` | Dual UR5e evaluation cell (default)   | nakul + sahadev UR5e |
| `face_robot_ur20`  | UR20 face interaction demo            | Single UR20          |

## Directory Structure

```
simforge_server_nvidia/
├── CMakeLists.txt                         # Gateway ROS2 package build
├── package.xml                            # ROS2 package manifest
├── docker-compose.yml                     # sim / prod profiles with GPU access
├── README.md
│
├── docker/
│   ├── Dockerfile                         # CUDA 12.8 + ROS2 Humble + cuRobo
│   └── ros_entrypoint.sh                  # Container entrypoint
│
├── config/                                # cuRobo configuration
│   ├── environments/                      # Environment selector YAMLs
│   │   ├── valid8_dual_ur5e.yaml
│   │   └── face_robot_ur20.yaml
│   ├── valid8_dual_ur5e/                  # Per-env cuRobo robot + world configs
│   │   ├── nakul_ur5e_curobo.yml
│   │   ├── sahadev_ur5e_curobo.yml
│   │   └── world_collision.yml
│   └── face_robot_ur20/
│       ├── ur20_curobo.yml
│       └── world_collision.yml
│
├── environments/                          # Per-environment ROS2 packages
│   ├── valid8_dual_ur5e/
│   │   ├── description/                   # URDF, meshes, rviz
│   │   └── control/                       # ros2_control config + launch
│   └── face_robot_ur20/
│       ├── description/                   # URDF, meshes
│       └── control/                       # ros2_control config + launch
│
├── launch/                                # Gateway-level launch files
│   ├── sim.launch.py                      # Simulation mode (mock hardware)
│   ├── real.launch.py                     # Real robot mode (UR driver)
│   └── gateway.launch.py                  # cuRobo command gateway
│
├── nodes/
│   └── command_gateway_curobo_node.py     # ROS2 node entry point
│
├── simforge_gateway_nvidia/               # Python package
│   ├── __init__.py
│   ├── config.py                          # Constants, dataclasses, env loading
│   ├── env_loader.py                      # Environment YAML config loader
│   ├── curobo_planner.py                  # cuRobo GPU motion planning
│   ├── collision_matrix.py                # Self-collision matrix generation
│   ├── trajectory_executor.py             # Trajectory dispatch (sim + real)
│   ├── rtde_controller.py                 # ur_rtde low-level wrapper
│   ├── joint_state_manager.py             # Mode-aware /joint_states
│   ├── protocol_executor.py               # Multi-pose protocol runner
│   └── rpc_handlers.py                    # WebSocket RPC implementations
│
├── scripts/
│   └── start_server.sh                    # Container supervisor script
│
├── tools/                                 # Development-only utilities
│   └── generate_self_collision_ignore.py   # ACM → cuRobo YAML converter
│
└── tests/
    └── test_curobo_gateway.py
```

## Quick Start

All commands should be run from `simforge_server_nvidia/`.

### Platform Profiles

| Profile | Platform | Mode | Command prefix |
|---|---|---|---|
| `sim` | x86_64 (NVIDIA GPU) | Simulation (mock HW) | `docker compose --profile sim` |
| `prod` | x86_64 (NVIDIA GPU) | Real robot (ur_rtde) | `docker compose --profile prod` |
| `jetson-sim` | Jetson Thor (aarch64) | Simulation (mock HW) | `sudo docker compose --profile jetson-sim` |
| `jetson-prod` | Jetson Thor (aarch64) | Real robot (ur_rtde) | `sudo docker compose --profile jetson-prod` |

> **Note:** On Jetson, Docker requires `sudo`. On x86_64 systems where your user is
> in the `docker` group, `sudo` is not needed.

### Starting a Container

The `ENV_CONFIG` variable selects the robot environment. Default is `valid8_dual_ur5e`.

```bash
# Dual UR5e cell — simulation (default environment)
docker compose --profile sim up -d --build

# Face robot UR20 — simulation
ENV_CONFIG=face_robot_ur20 docker compose --profile sim up -d --build

# Jetson Thor — dual UR5e simulation (default)
sudo docker compose --profile jetson-sim up -d --build

# Jetson Thor — face robot UR20 simulation
sudo ENV_CONFIG=face_robot_ur20 docker compose --profile jetson-sim up -d --build
```

### Stopping a Container

Stop whichever profile you started — you **must** specify the same profile:

```bash
# x86_64
docker compose --profile sim down

# Jetson
sudo docker compose --profile jetson-sim down
```

### Switching Environments

To switch from one environment to another, stop the running container first,
then start with the new `ENV_CONFIG`:

```bash
# Stop the current environment
sudo docker compose --profile jetson-sim down

# Start with a different environment
sudo ENV_CONFIG=face_robot_ur20 docker compose --profile jetson-sim up -d
```

> **Tip:** You can check which environment a running container is using:
> ```bash
> sudo docker exec simforge-server-nvidia-jetson-sim printenv ENV_CONFIG
> ```

### Real Robot Mode

```bash
# Dual UR5e (x86_64)
ROBOT_IP_NAKUL_UR5E=192.168.1.9 \
ROBOT_IP_SAHADEV_UR5E=192.168.1.16 \
  docker compose --profile prod up -d --build

# Face robot UR20 (x86_64)
ENV_CONFIG=face_robot_ur20 ROBOT_IP_UR20=10.0.0.1 \
  docker compose --profile prod up -d --build

# Jetson — dual UR5e
sudo ROBOT_IP_NAKUL_UR5E=192.168.1.9 \
     ROBOT_IP_SAHADEV_UR5E=192.168.1.16 \
  docker compose --profile jetson-prod up -d --build

# Jetson — face robot UR20
sudo ENV_CONFIG=face_robot_ur20 ROBOT_IP_UR20=10.0.0.1 \
  docker compose --profile jetson-prod up -d --build
```

### Viewing Logs

```bash
# Follow live logs
sudo docker logs -f simforge-server-nvidia-jetson-sim

# Last 50 lines
sudo docker logs --tail 50 simforge-server-nvidia-jetson-sim
```

### Running Tests

```bash
# Simulation-only
python tests/test_curobo_gateway.py --sim-only

# Full sim → real
python tests/test_curobo_gateway.py --server <SERVER_IP>
```

## GPU Requirements

- **x86_64**: NVIDIA GPU with CUDA 12.2+ support, NVIDIA Container Toolkit
- **Jetson Thor**: JetPack 7.0 with CUDA 13.x, NVIDIA Container Toolkit (default-runtime: nvidia)
- Tested on NVIDIA RTX Pro 6000 (x86_64) and NVIDIA Jetson AGX Thor (aarch64)

## Configuration

### Velocity Scaling

```bash
MAX_VELOCITY_SCALING=0.15   # Slower (safer)
MAX_VELOCITY_SCALING=0.3    # Faster
```

### cuRobo Interpolation

```bash
INTERPOLATION_DT=0.02   # 50 Hz waypoints (default)
INTERPOLATION_DT=0.01   # 100 Hz waypoints (smoother)
```

## Adding a New Environment

1. Create a YAML in `config/environments/<env_name>.yaml`
2. Add cuRobo configs under `config/<env_name>/`
3. Create description + control ROS2 packages under `environments/<env_name>/`
4. Add the control package to the `control_package_map` in `launch/sim.launch.py`
5. Add environment branching in `launch/real.launch.py`
6. Add `COPY` lines in `docker/Dockerfile`

## Architecture

```
┌──────────────┐     WebSocket :8766     ┌──────────────────────────┐
│ simforge_     │ ◄─────────────────────►│ command_gateway_curobo   │
│ client (GUI)  │                        │                          │
└──────────────┘                         │  rpc_handlers            │
                                         │       │                  │
┌──────────────┐     WebSocket :9090     │  curobo_planner (GPU)    │
│ Foxglove     │ ◄── foxglove_bridge     │       │                  │
│ Studio       │                         │  trajectory_executor     │
└──────────────┘                         │    ┌──────┴──────┐       │
                                         │    │             │       │
                                         │  (sim)        (real)     │
                                         └────┼─────────────┼───────┘
                                              │             │
                                      FollowJoint     ur_rtde servoJ
                                      Trajectory       (500 Hz RTDE)
                                              │
                                    ┌─────────▼──────────┐
                                    │ ros2_control_node  │
                                    │ (UR driver / mock) │
                                    └────────────────────┘
```
