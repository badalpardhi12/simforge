# simforge_client

Cross-platform WebSocket client for controlling robots via the SimForge server stack. Designed for macOS (Apple Silicon) but works on any platform with Python 3.10+.

**Does NOT require ROS 2** — pure WebSocket communication.

## Features

- **WebSocket Client** (`command_client.py`): JSON-RPC connection to the Command Gateway with automatic reconnection, heartbeat, and E-Stop
- **Safety Monitor** (`safety_monitor.py`): Independent heartbeat / connection monitoring and keyboard E-Stop
- **Proto-Sim GUI** (`gui/proto_sim_client.py`): wxPython interface for multi-pose protocol simulation — generates poses locally and sends them to the server
- **Foxglove Bridge** (`foxglove_client.py`): Optional Foxglove WebSocket integration for visualisation
- **Pose Generation** (`utils/pose_generation.py`): Spherical-coordinate pose sampling matching the Genesis simulator parameters

## Package Structure

```
simforge_client/
├── __init__.py            # SimforgeClient + SafetyMonitor exports
├── command_client.py      # Core WebSocket JSON-RPC client
├── foxglove_client.py     # Optional Foxglove bridge
├── run_client.py          # Entry point launcher
├── safety_monitor.py      # Heartbeat & E-Stop monitor
├── gui/
│   └── proto_sim_client.py  # wxPython protocol simulation UI
├── utils/
│   └── pose_generation.py   # Spherical pose sampling
└── tests/
    ├── test_move.py
    ├── test_face_link_mode_switch.py
    └── test_vla.py
```

## Installation

```bash
cd simforge          # repository root
pip install -e simforge_client                   # core only
pip install -e "simforge_client[gui]"            # + wxPython GUI
pip install -e "simforge_client[foxglove]"       # + Foxglove bridge
pip install -e "simforge_client[full]"           # everything
pip install -e "simforge_client[dev]"            # + pytest, mypy
```

## Usage

### CLI

```bash
# Connect to server and open the proto-sim GUI
simforge-proto-sim --server 192.168.1.12 --port 8766

# Or run as a module
python -m simforge_client --server 192.168.1.12 --port 8766
```

### Python API

```python
import asyncio
from simforge_client import SimforgeClient

async def main():
    client = SimforgeClient(server="192.168.1.12", port=8766)
    await client.connect()

    # Move a robot to joint positions
    result = await client.move_joints("nakul", [0, -1.57, 0, -1.57, 0, 0])

    # Move to a Cartesian pose
    result = await client.move_cartesian("nakul", position=[0.4, 0.0, 0.3],
                                          orientation=[0, 0, 0, 1])

    # Emergency stop
    await client.estop()

asyncio.run(main())
```

## Configuration

The client connects to whichever SimForge server is running — either:

| Server | Typical host | Port |
|---|---|---|
| **simforge_genesis** (Genesis sim) | localhost | 8766 |
| **simforge_server_nvidia** (cuRobo / real robot) | Jetson / workstation IP | 8766 |

Both servers expose the same JSON-RPC WebSocket protocol so the client is interchangeable.

## License

MIT
