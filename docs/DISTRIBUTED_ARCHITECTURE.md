# Simforge Distributed Architecture

A distributed robotics control system using ROS 2 Humble with Foxglove WebSocket for cross-platform communication.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              NETWORK (WebSocket + DDS)                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │                                                           │
         ▼                                                           ▼
┌─────────────────────────┐                          ┌─────────────────────────────────┐
│    MAC STUDIO (Client)  │                          │   JETSON THOR / WORKSTATION     │
│                         │                          │          (Server)               │
│  ┌───────────────────┐  │    WebSocket:9090        │  ┌─────────────────────────────┐│
│  │  Foxglove Studio  │◄─┼──────────────────────────┼──│   Foxglove Bridge Node      ││
│  │  (Visualization)  │  │                          │  └─────────────────────────────┘│
│  └───────────────────┘  │                          │                                 │
│                         │    WebSocket:8765        │  ┌─────────────────────────────┐│
│  ┌───────────────────┐  │                          │  │   Command Gateway Node      ││
│  │  simforge_client  │◄─┼──────────────────────────┼──│   (Action Server wrapper)   ││
│  │  (Python + UI)    │  │                          │  └─────────────────────────────┘│
│  └───────────────────┘  │                          │                                 │
└─────────────────────────┘                          │  ┌─────────────────────────────┐│
                                                     │  │ Robot Control Stack         ││
                                                     │  │ - Safety Watchdog           ││
                                                     │  │ - Robot Control (ur_rtde)   ││
                                                     │  │ - Perception (nvblox)       ││
                                                     │  │ - VLA Inference             ││
                                                     │  │ - Orchestrator              ││
                                                     │  └─────────────────────────────┘│
                                                     └─────────────────────────────────┘
```

## Quick Start

### 1. Server (AI Workstation)

```bash
# Build Docker image
docker compose --profile dev build

# Start server stack
docker compose --profile dev up

# Or without Docker (requires ROS 2 Humble installed):
cd simforge
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 launch simforge_server full_stack.launch.py simulation_mode:=true
```

### 2. Client (Mac Studio)

```bash
# Install client package
cd simforge_client
pip install -e ".[full]"

# Connect and test
python tests/test_move.py --server <WORKSTATION_IP>
```

### 3. Visualization (Foxglove Studio)

1. Download [Foxglove Studio](https://foxglove.dev/download) for macOS
2. Open Foxglove Studio
3. Connect to: `ws://<WORKSTATION_IP>:9090`
4. Visualize robot state, camera feeds, and planned trajectories

## Directory Structure

```
simforge/
├── msgs/                       # Custom ROS 2 message definitions
│   └── simforge_msgs/
│       ├── msg/                # Messages (Heartbeat, RobotState, SafetyStatus)
│       ├── srv/                # Services (GetVLAAction, GetRobotState)
│       └── action/             # Actions (MoveRobot, ExecuteTask)
├── simforge_client/            # Mac client (pure Python, no ROS 2)
│   ├── command_client.py       # WebSocket client for commands
│   ├── safety_monitor.py       # Local safety monitoring
│   ├── foxglove_client.py      # Foxglove visualization client
│   └── tests/                  # Test scripts
├── simforge_server/            # ROS 2 server nodes
│   ├── nodes/
│   │   ├── safety_watchdog_node.py    # CRITICAL: Safety monitoring
│   │   ├── command_gateway_node.py    # WebSocket <-> ROS 2 bridge
│   │   ├── robot_control_node.py      # UR robot control via RTDE
│   │   ├── perception_node.py         # Camera/nvblox integration
│   │   ├── vla_inference_node.py      # VLA model inference
│   │   └── orchestrator_node.py       # High-level task coordination
│   └── launch/
│       ├── full_stack.launch.py       # Launch all components
│       ├── robot_bringup.launch.py    # Minimal robot control
│       ├── perception.launch.py       # Perception pipeline
│       └── vla.launch.py              # VLA inference
├── config/
│   ├── simforge_config.yaml    # Main configuration
│   ├── cumotion/               # cuMotion configs
│   └── fastdds.xml             # DDS configuration
└── docker/
    ├── Dockerfile.server.x86   # AI Workstation
    ├── Dockerfile.server.l4t   # Jetson Thor
    └── docker-compose.yml      # Service orchestration
```

## Safety Architecture

⚠️ **CRITICAL**: The safety system is the foundation of this architecture.

### Heartbeat Protocol
- Client sends heartbeat every 20ms (50Hz)
- Server expects heartbeat within 100ms timeout
- 3 consecutive misses → protective stop

### Emergency Stop Chain
```
Mac E-Stop → WebSocket → Safety Watchdog → Robot stopJ(2.0)
                                ↓
                        cuMotion abort_motion()
```

## Configuration

### Environment Variables (Server)

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fastdds.xml
```

### Robot Configuration

Edit `config/simforge_config.yaml`:

```yaml
robots:
  ur20:
    ip: "192.168.1.10"
    name: "ur20"
    max_velocity_scale: 1.0
    default_velocity_scale: 0.5
```

## Usage Examples

### Python Client

```python
import asyncio
from simforge_client import SimforgeClient

async def main():
    async with SimforgeClient(server_ip="192.168.1.100") as client:
        # Move to joint position
        result = await client.move_robot(
            "ur20",
            target_joints=[0, -1.57, 1.57, 0, 0, 0],
            velocity_scale=0.3
        )
        print(f"Move result: {result.success}")
        
        # Execute VLA task
        result = await client.execute_task(
            instruction="pick up the red cube",
            robot_name="ur20"
        )
        print(f"Task result: {result.success}")

asyncio.run(main())
```

### ROS 2 Launch

```bash
# Full stack (all components)
ros2 launch simforge_server full_stack.launch.py robot_ip:=192.168.1.10

# Robot only (no VLA/perception)
ros2 launch simforge_server robot_bringup.launch.py robot_ip:=192.168.1.10

# Simulation mode
ros2 launch simforge_server full_stack.launch.py simulation_mode:=true
```

## Development

### Building Messages

```bash
cd simforge
colcon build --packages-select simforge_msgs
source install/setup.bash
```

### Testing

```bash
# Server tests (requires ROS 2)
colcon test --packages-select simforge_server

# Client tests (pure Python)
cd simforge_client
pytest tests/
```

## Deployment

### AI Workstation (Development)

```bash
docker compose --profile dev up -d
```

### Jetson Thor (Production)

```bash
docker compose --profile prod up -d
```

## Troubleshooting

### Connection Issues

1. Check network connectivity: `ping <SERVER_IP>`
2. Verify Foxglove Bridge: `curl http://<SERVER_IP>:9090`
3. Check ROS 2 topics: `ros2 topic list`

### Safety Stop Triggered

1. Check heartbeat: Ensure client is running and connected
2. Check force limits: Verify no excessive force on TCP
3. Reset: `ros2 service call /safety/reset std_srvs/srv/Trigger`

### VLA Inference Issues

1. Check GPU memory: `nvidia-smi`
2. Verify model loaded: Check `/vla/status` topic
3. Test with simulation mode first

## License

MIT License - See LICENSE file for details.
