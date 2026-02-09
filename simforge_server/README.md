# Simforge Server

ROS 2 application layer that bridges a WebSocket-based client GUI to the UR robot control stack (UR driver + MoveIt 2). The entire server runs inside a Docker container.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Docker Container (simforge_server_dev)                            │
│                                                                     │
│  full_stack.launch.py launches:                                     │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │  UR Robot       │  │  MoveIt 2    │  │  Foxglove Bridge     │    │
│  │  Driver         │  │  move_group  │  │  (visualization)     │    │
│  │                 │  │              │  │  port 9090           │    │
│  │  - RTDE comms   │  │  - IK        │  └──────────────────────┘    │
│  │  - controllers  │  │  - Planning  │                              │
│  │  - headless     │  │  - Collision │  ┌──────────────────────┐    │
│  │    mode         │  │    checking  │  │  Command Gateway     │    │
│  └───────┬─────────┘  └──────┬───────┘  │  (this package)      │    │
│          │                   │          │  WebSocket :8766      │◄───── Client
│          │    ROS 2 Topics   │          │                      │    │
│          │    & Services     │          │  - JSON-RPC over WS  │    │
│          └───────────────────┘          │  - IK solving        │    │
│                                         │  - Trajectory exec   │    │
│                                         │  - Mode switching    │    │
│                                         └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
          │                                         ▲
          │ RTDE (port 30004)                       │ WebSocket
          ▼                                         │
    ┌───────────┐                           ┌───────────────┐
    │  UR5e     │                           │  Client GUI   │
    │  Robot    │                           │  (wxPython)   │
    │  .9       │                           │  macOS/.12    │
    └───────────┘                           └───────────────┘
```

## File Structure

```
simforge_server/
├── CMakeLists.txt                              # Colcon build — installs command_gateway_node.py
├── package.xml                                 # ROS 2 package manifest
├── nodes/
│   └── command_gateway_node.py                 # The single ROS node (~3000 lines)
└── simforge_server/                            # Python package (ament_python_install_package)
    └── utils/
        └── pose_generation.py                  # Spherical pose sampling geometry
```

## How It Starts

1. `docker compose --profile dev up` starts the container using `docker-compose.yml`
2. The entrypoint runs `ros2 launch valid8_cell_control full_stack.launch.py`
3. That launch file starts five components:
   - **UR Robot Driver** (`ur_control.launch.py`) — RTDE connection to the robot, `scaled_joint_trajectory_controller`, `joint_state_broadcaster`
   - **MoveIt move_group** (`move_group.launch.py`) — motion planning, IK, collision checking
   - **Foxglove Bridge** — ROS topic visualization over WebSocket (port 9090)
   - **Command Gateway** (`command_gateway_node.py`) — this package, WebSocket on port 8766
   - **Headless Keepalive** — monitors `robot_program_running` and resends the URScript if it drops

The source directories are volume-mounted into the container for hot-reload during development. Editing files on the host takes effect on container restart — no image rebuild needed.

## Command Gateway Node

`command_gateway_node.py` is the single ROS 2 node in this package. It is a `rclpy.Node` subclass that runs an asyncio WebSocket server alongside the ROS 2 spin loop.

### Responsibilities

| Responsibility | How |
|---|---|
| Accept client connections | `websockets.serve()` on `0.0.0.0:8766`, up to 5 clients |
| Route JSON messages | `process_message()` dispatches by `type` field |
| Solve inverse kinematics | Calls MoveIt `/compute_ik` service |
| Plan collision-free trajectories | Calls MoveIt `/plan_kinematic_path` service |
| Execute on real robot | Sends `RobotTrajectory` via MoveIt `/execute_trajectory` action |
| Execute in simulation | Publishes interpolated `JointState` messages on `/joint_states` |
| Switch execution modes | Activates/deactivates the UR driver's `joint_state_broadcaster` controller |
| Monitor robot health | Subscribes to `/io_and_status_controller/robot_program_running` |
| Recovery from failures | Resends robot program, retries trajectories (up to 2 retries per pose) |

### ROS 2 Interfaces Used

**Subscriptions:**
- `/joint_states` (`sensor_msgs/JointState`) — real robot joint positions from `joint_state_broadcaster`
- `/io_and_status_controller/robot_program_running` (`std_msgs/Bool`) — UR driver program state

**Publishers:**
- `/joint_states` (`sensor_msgs/JointState`) — simulated joint positions (only in simulation mode)
- `/safety/heartbeat` (`std_msgs/String`) — forwarded client heartbeats
- `/safety/emergency_stop` (`std_msgs/String`) — e-stop broadcast

**Service Clients:**
- `/compute_ik` (`moveit_msgs/GetPositionIK`) — MoveIt inverse kinematics
- `/plan_kinematic_path` (`moveit_msgs/GetMotionPlan`) — MoveIt trajectory planning with OMPL
- `/get_planning_scene` (`moveit_msgs/GetPlanningScene`) — collision environment
- `/io_and_status_controller/resend_robot_program` (`std_srvs/Trigger`) — restart URScript

**Action Clients:**
- `/execute_trajectory` (`moveit_msgs/ExecuteTrajectory`) — **primary** real robot execution path (MoveIt → controller)
- `/scaled_joint_trajectory_controller/follow_joint_trajectory` (`control_msgs/FollowJointTrajectory`) — fallback execution path

### Key Internal Methods

| Method | Purpose |
|---|---|
| `_solve_ik_for_pose()` | Solve IK via MoveIt, seed with current joint positions for continuity |
| `_normalize_joint_angles()` | Wrap IK result joints to nearest 2π offset of seed (avoids unnecessary rotations) |
| `_plan_motion_moveit_sync()` | Plan trajectory with OMPL + TOTG time parameterization, enforce velocity/acceleration limits |
| `_execute_real_robot_trajectory()` | Send `RobotTrajectory` via MoveIt `ExecuteTrajectory` action, wait for completion |
| `_execute_trajectory()` | Simulate trajectory by interpolating and publishing `JointState` at 50 Hz |
| `_set_joint_state_broadcaster()` | Activate/deactivate the UR driver's `joint_state_broadcaster` via `ros2 control switch_controller` |
| `_ensure_robot_ready()` | Check `robot_program_running`, resend program if needed, wait for controller activation |
| `_publish_joint_state()` | Publish simulated joint state (gated — only runs in simulation mode) |

## Client ↔ Server Communication Protocol

### Transport

JSON messages over a persistent WebSocket connection (`ws://<server_ip>:8766`).

The client (`simforge_client/command_client.py` → `SimforgeClient`) establishes a single WebSocket connection and multiplexes all communication through it. Messages are matched by `request_id`.

### Message Flow

```
Client                              Server
  │                                    │
  │─── {"type": "rpc", ...} ──────────►│  Request
  │                                    │
  │◄── {"type": "rpc_feedback", ...} ──│  Streaming feedback (for long-running ops)
  │◄── {"type": "rpc_feedback", ...} ──│
  │                                    │
  │◄── {"type": "rpc_result", ...} ────│  Final result
  │                                    │
  │─── {"type": "heartbeat", ...} ────►│  Periodic (2 Hz)
  │─── {"type": "ping", ...} ─────────►│  Latency check
  │◄── {"type": "result", ...} ────────│  Pong
```

### Message Types

**Client → Server:**

| `type` | Purpose |
|---|---|
| `rpc` | Call a server method (see RPC Methods below) |
| `heartbeat` | Keep-alive signal (forwarded to safety watchdog) |
| `ping` | Latency measurement |
| `emergency_stop` | Immediate stop, broadcast to all clients |
| `move_robot` | Direct move command (legacy) |

**Server → Client:**

| `type` | Purpose |
|---|---|
| `rpc_result` | Response to an `rpc` call |
| `rpc_feedback` | Streaming progress during long-running operations |
| `result` | Response to `ping` / `move_robot` |
| `error` | Error response |
| `emergency_stop_active` | Broadcast when any client triggers e-stop |

### RPC Methods

All RPC calls use the same envelope:

```json
{
    "type": "rpc",
    "request_id": "mac_client_42",
    "method": "<method_name>",
    "params": { ... }
}
```

#### `get_environment_info`

Returns available robots, target objects, and their TF transforms relative to `base_link`.

```json
// Response
{
    "success": true,
    "robots": ["nakul_ur5e"],
    "objects": ["face_link", "table_link", "shop_floor"],
    "object_transforms": {
        "face_link": {
            "position": [0.3, 0.0, 0.5],
            "orientation": [0.0, 0.0, 0.0, 1.0]
        }
    },
    "reference_frame": "base_link"
}
```

#### `get_robot_status`

Checks real robot connectivity — UR driver, MoveIt, robot program state. Used by the GUI to show a connection indicator.

```json
// Response
{
    "success": true,
    "real_robot_available": true,
    "simulation_available": true,
    "available_modes": ["simulation", "real", "both"],
    "connection_details": {
        "follow_trajectory_action": "available",
        "execute_trajectory_action": "available",
        "moveit": "available",
        "robot_program_running": true
    },
    "current_joint_positions": [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
}
```

#### `prepare_mode`

**Must be called before `run_proto_sim`.** Switches the server between simulation and real-robot modes:

- **Simulation mode:** deactivates `joint_state_broadcaster` so only the gateway publishes simulated `/joint_states`
- **Real/both mode:** activates `joint_state_broadcaster`, validates UR driver + MoveIt + robot program

```json
// Request
{ "mode": "simulation" | "both" }

// Response
{
    "ready": true,
    "message": "Robot ready for both mode\n✓ Joint-state broadcaster active\n✓ UR Driver\n✓ MoveIt\n✓ Robot Program Running"
}
```

#### `run_proto_sim`

Runs the full protocol simulation loop. This is a **long-running operation** — it sends streaming `rpc_feedback` messages for each pose, then a final `rpc_result`.

```json
// Request
{
    "robot_name": "nakul_ur5e",
    "mode": "simulation" | "both",
    "move_speed": 0.3,
    "idle_time": 2.0,
    "poses": [
        {
            "name": "H0_V0_D350_R-90_P15_Y0",
            "position": [0.25, 0.0, 0.45],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "parameters": {"horiz": 0, "vert": 0, "distance": 350, "roll": -90, "pitch": 15, "yaw": 0}
        }
    ]
}
```

**Feedback messages** (one per pose):
```json
{
    "type": "rpc_feedback",
    "request_id": "...",
    "current_pose_index": 0,
    "total_poses": 3,
    "current_pose_name": "H0_V0_D350_R-90_P15_Y0",
    "status": "moving" | "idle" | "collision_rejected" | "ik_failed" | "real_robot_failed"
}
```

**Final result:**
```json
{
    "type": "rpc_result",
    "success": true,
    "completed": 3,
    "total": 3,
    "collision_rejected": 0,
    "ik_failed": 0,
    "real_failed": 0
}
```

#### `stop_proto_sim`

Requests early termination of a running protocol simulation. The loop checks `_proto_sim_stop_requested` between poses.

## Proto-Sim Execution Pipeline

This is the core workflow when `run_proto_sim` is called:

```
For each pose in the list:
│
├─ 1. Solve IK
│     ├─ Seed: real_robot_joint_positions (real/both) or sim_joint_positions (sim)
│     ├─ Call MoveIt /compute_ik
│     └─ Normalize joint angles to nearest 2π offset of seed
│
├─ 2. Plan trajectory
│     ├─ Start state: real robot position (real/both) or sim position (sim)
│     ├─ Call MoveIt /plan_kinematic_path (OMPL RRTConnect)
│     ├─ MoveIt applies TOTG time parameterization
│     └─ Enforce velocity limits (max 1.0 rad/s shoulder, 1.5 rad/s wrist)
│
├─ 3. Drift check (real/both only)
│     ├─ Re-read real robot state
│     └─ If drifted > 3° during planning → re-plan from fresh state
│
├─ 4. Execute
│     ├─ Real/Both: send RobotTrajectory via /execute_trajectory action
│     │   ├─ On failure: wait 5s → ensure_robot_ready → re-plan → retry (up to 2x)
│     │   └─ On success: sync sim_joint_positions to target
│     └─ Simulation: interpolate trajectory, publish JointState at 50 Hz
│
├─ 5. Idle at pose (configurable dwell time)
│
└─ 6. Send feedback to client
```

After all poses: plan and execute return-home trajectory (with retry logic).

## Mode Switching

The server supports two execution modes (selected by the client GUI):

| Mode | `joint_state_broadcaster` | Gateway publishes `/joint_states`? | Robot moves? |
|---|---|---|---|
| **Simulation** | Deactivated | Yes (simulated interpolation) | No |
| **Both** (Sim + Real) | Active | No | Yes |

Mode switching is handled by `rpc_prepare_mode()` → `_set_joint_state_broadcaster()`, which calls `ros2 control switch_controller` to activate/deactivate the UR driver's joint state broadcaster. This prevents dual-publisher conflicts on `/joint_states`.

## Pose Generation

`simforge_server/utils/pose_generation.py` implements spherical coordinate pose sampling:

- **Input:** parameter ranges (horizontal offset, vertical offset, distance, roll, pitch, yaw)
- **Output:** list of `ProtoPose` objects with position (meters) and orientation (quaternion) in the target object's local frame

The client transforms these poses into `base_link` frame using TF data from `get_environment_info`, then sends the pre-computed world-frame poses to `run_proto_sim`.

## Development

### Hot-reload workflow

Source files are volume-mounted into the container. To pick up changes:

```bash
# Restart the container (picks up Python changes immediately)
docker compose --profile dev restart ros2_server_dev

# Check logs
docker logs -f simforge_server_dev 2>&1 | grep command_gateway
```

### Adding a new RPC method

1. Add `async def rpc_your_method(self, params)` to `CommandGatewayNode`
2. Register it in `handle_rpc()`:
   ```python
   elif method == 'your_method':
       result = await self.rpc_your_method(params)
   ```
3. Call from client: `await client.call_rpc("your_method", {"key": "value"})`

### Adding a new message type

1. Add a handler `async def handle_your_type(self, client, msg)` to `CommandGatewayNode`
2. Register in `process_message()`:
   ```python
   elif msg_type == 'your_type':
       await self.handle_your_type(client, msg)
   ```

### Key configuration

| Parameter | Default | Set via |
|---|---|---|
| WebSocket port | 8766 | Launch file `websocket_port` arg |
| Max velocity (shoulder/elbow) | 1.0 rad/s | `valid8_cell_moveit_config/config/joint_limits.yaml` |
| Max velocity (wrist) | 1.5 rad/s | Same file |
| Path tolerance | 0.5 rad | `valid8_cell_control/config/ur_controllers.yaml` |
| Speed cap (real mode) | 0.3 | Hardcoded in `rpc_run_proto_sim()` |
| Trajectory retries | 2 | Hardcoded in proto-sim pose loop |
