# SimForge

Monorepo for the SimForge robotics platform — simulation, GPU-accelerated motion planning, and cross-platform client control for Universal Robots workcells.

## Packages

| Package | Description | Runtime |
|---|---|---|
| [**simforge_genesis**](simforge_genesis/) | Genesis physics simulator with multi-robot joint/Cartesian control, IK, OMPL planning, and collision checking | Python 3.10+, Genesis, PyTorch |
| [**simforge_client**](simforge_client/) | Cross-platform WebSocket client + wxPython GUI for protocol simulation and robot operation (no ROS 2 required) | Python 3.10+ |
| [**simforge_server_nvidia**](simforge_server_nvidia/) | GPU-accelerated motion-planning server using NVIDIA cuRobo, with real-robot control via ur_rtde and sim via ros2_control | ROS 2 Jazzy, CUDA 12+, Docker |

### Supporting Data

| Directory | Description |
|---|---|
| [**macara_plans/**](macara_plans/) | Multi-pose protocol plan files (JSON) used by the proto-sim workflow |

## Architecture Overview

```
┌─────────────────────┐         WebSocket :8766         ┌───────────────────────────┐
│                     │ ◄──────────────────────────────► │  simforge_server_nvidia   │
│   simforge_client   │                                  │  (cuRobo GPU planning     │
│   (macOS / any OS)  │  ┌─────────────────────────┐     │   + ur_rtde real control) │
│                     │  │  simforge_genesis        │     └───────────────────────────┘
└─────────────────────┘  │  (Genesis simulation     │               │
         │               │   + OMPL planning)       │         FollowJointTrajectory
         │               └──────────┬───────────────┘         or servoJ (500 Hz)
         │                          │                               │
         │               WebSocket :8766                 ┌──────────▼──────────┐
         └──────────────────────────┘                    │  UR Robot / mock HW │
                                                         └─────────────────────┘
```

The **client** connects to either the Genesis simulator *or* the NVIDIA cuRobo server — both expose the same JSON-RPC WebSocket protocol on port 8766.

## Quick Start

### 1. Genesis Simulator (local development)

```bash
# Install simforge_genesis in dev mode
pip install -e ".[dev]"

# Launch the proto-sim workspace
simforge_genesis proto_sim --config simforge_genesis/environment/presets/face_robot.yaml
```

### 2. cuRobo Server (Jetson / workstation with GPU)

```bash
cd simforge_server_nvidia

# Simulation mode (mock hardware)
docker compose --profile sim up --build

# Real robot mode
ROBOT_IP_NAKUL_UR5E=192.168.1.9 \
ROBOT_IP_SAHADEV_UR5E=192.168.1.16 \
  docker compose --profile prod up --build
```

### 3. Client (macOS or any platform)

```bash
pip install -e simforge_client[gui]

# Connect to the server
simforge-proto-sim --server <SERVER_IP> --port 8766
```

## Development

```bash
# Format & lint
ruff format .
ruff check . --fix

# Run tests (Genesis package)
pytest simforge_genesis/tests/

# Run tests (client)
pytest simforge_client/tests/
```

## License

Proprietary
