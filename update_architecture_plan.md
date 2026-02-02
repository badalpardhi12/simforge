# Simforge Distributed Architecture Plan
**Target System:** Mac Studio (Control) <-> Jetson Thor (Edge Compute)  
**Dev System:** Mac Studio (Control) <-> AI Workstation (Simulated Edge)  
**Last Updated:** Document enhanced with deep research validation (2025)

---

## 0. Executive Summary & Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Middleware | ROS 2 Humble | LTS until 2027, Isaac ROS compatibility, ur_ros_rtde support |
| macOS Client | Foxglove WebSocket (NOT native ROS 2) | No official ROS 2 Apple Silicon support; Foxglove Bridge v0.8.2 is high-performance C++ |
| GPU Motion Planning | cuMotion via MoveIt 2 plugin | 90%+ collision-free success rate, nvblox ESDF integration |
| UR Robot Driver | ur_ros_rtde | Active development (2025), action servers for MoveJ/MoveL, MoveIt2 integration |
| Perception | nvblox (TSDF/ESDF) | Sub-millisecond ESDF on Thor (1.0ms), dynamic reconstruction support |
| VLA Architecture | Dual-system (Fast-in-Slow pattern) | High-level VLA reasoning + low-frequency; separate real-time control loop |

---

## 1. Architectural Overview

We are refactoring `simforge` from a monolithic script into a distributed **Client-Server** architecture using **ROS 2** as the middleware with **Foxglove WebSocket** for cross-platform communication.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              NETWORK (WebSocket + DDS)                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
         │                                                           │
         ▼                                                           ▼
┌─────────────────────────┐                          ┌─────────────────────────────────┐
│    MAC STUDIO (Client)  │                          │   JETSON THOR / WORKSTATION     │
│                         │                          │          (Server)               │
│  ┌───────────────────┐  │                          │  ┌─────────────────────────────┐│
│  │  Foxglove Studio  │  │    WebSocket:9090        │  │   Foxglove Bridge Node      ││
│  │  (Visualization)  │◄─┼──────────────────────────┼──│   (ros2_foxglove_bridge)    ││
│  └───────────────────┘  │                          │  └─────────────────────────────┘│
│                         │                          │                                 │
│  ┌───────────────────┐  │    WebSocket:8765        │  ┌─────────────────────────────┐│
│  │  simforge_client  │◄─┼──────────────────────────┼──│   Command Gateway Node      ││
│  │  (Python + UI)    │  │    (Custom JSON-RPC)     │  │   (Action Server wrapper)   ││
│  └───────────────────┘  │                          │  └─────────────────────────────┘│
│                         │                          │               │                 │
│  ┌───────────────────┐  │                          │               ▼                 │
│  │  Safety Monitor   │  │    Heartbeat:50Hz        │  ┌─────────────────────────────┐│
│  │  (Watchdog)       │──┼──────────────────────────┼─►│   Safety Watchdog Node      ││
│  └───────────────────┘  │                          │  └─────────────────────────────┘│
└─────────────────────────┘                          │               │                 │
                                                     │               ▼                 │
                                                     │  ┌─────────────────────────────┐│
                                                     │  │ Robot Control Stack         ││
                                                     │  │ - ur_ros_rtde               ││
                                                     │  │ - MoveIt 2 + cuMotion       ││
                                                     │  │ - nvblox (ESDF)             ││
                                                     │  │ - VLA Inference             ││
                                                     │  └─────────────────────────────┘│
                                                     └─────────────────────────────────┘
```

*   **The Client (Mac Studio):** Runs the "Commander". Captures user intent (text/mouse), visualizes state via Foxglove Studio, sends high-level requests via WebSocket.
*   **The Server (AI Workstation / Jetson Thor):** Runs the "Brain". Handles Perception (nvblox), Motion Planning (cuMotion), Robot Control (ur_ros_rtde), and Intelligence (VLA).

---

## 1.1 Safety Architecture (CRITICAL - Implement First)

**Rationale:** Remote robot control without safety mechanisms is dangerous. This must be the foundation.

### 1.1.1 Heartbeat Protocol
```python
# Client sends heartbeat every 20ms (50Hz)
# Server expects heartbeat within 100ms timeout
# If 3 consecutive heartbeats missed → trigger protective stop

message Heartbeat {
    uint64 timestamp_ns
    uint32 sequence_number
    string client_id
}
```

### 1.1.2 Safety Watchdog Node (Server-side)
```python
class SafetyWatchdogNode(Node):
    def __init__(self):
        self.last_heartbeat = time.time()
        self.heartbeat_timeout = 0.1  # 100ms
        self.consecutive_misses = 0
        self.max_misses = 3
        
        # Subscribe to heartbeat
        self.create_subscription(Heartbeat, '/safety/heartbeat', self.heartbeat_cb, 10)
        
        # Timer to check heartbeat (100Hz)
        self.create_timer(0.01, self.check_heartbeat)
        
        # Service client to robot driver
        self.stop_client = self.create_client(Trigger, '/robot/protective_stop')
    
    def check_heartbeat(self):
        if time.time() - self.last_heartbeat > self.heartbeat_timeout:
            self.consecutive_misses += 1
            if self.consecutive_misses >= self.max_misses:
                self.trigger_protective_stop()
                self.get_logger().error("SAFETY: Connection lost - protective stop triggered")
```

### 1.1.3 Emergency Stop Chain
```
Mac E-Stop Button → WebSocket → Safety Watchdog → ur_ros_rtde → Robot stopJ(2.0)
                                      ↓
                              cuMotion abort_motion()
```

---

## 2. Phase 1: Infrastructure & Networking (Weeks 1-2)

**Goal:** Establish a stable communication channel between Mac and Workstation using Docker, ROS 2, and Foxglove Bridge.

### 2.1 Repository Restructuring (Monorepo)
```text
simforge/
├── msgs/                       # Custom ROS 2 .msg, .srv, .action files
│   ├── simforge_msgs/
│   │   ├── msg/
│   │   │   ├── Heartbeat.msg
│   │   │   └── RobotState.msg
│   │   ├── srv/
│   │   │   └── GetVLAAction.srv
│   │   └── action/
│   │       └── MoveRobot.action
│   └── CMakeLists.txt
├── simforge_client/            # (Mac) Python Client & WebSocket
│   ├── foxglove_client.py
│   ├── command_client.py
│   └── safety_monitor.py
├── simforge_server/            # (Workstation/Jetson) ROS 2 Nodes
│   ├── launch/
│   ├── nodes/
│   │   ├── command_gateway_node.py
│   │   ├── safety_watchdog_node.py
│   │   ├── perception_node.py
│   │   └── vla_inference_node.py
│   └── wrappers/
│       ├── cumotion_wrapper.py
│       └── ur_rtde_wrapper.py
├── docker/
│   ├── Dockerfile.server.x86   # AI Workstation (dev)
│   ├── Dockerfile.server.l4t   # Jetson Thor (production)
│   └── Dockerfile.client       # Mac (optional, mostly native)
├── config/
│   ├── robots/                 # URDF + XRDF files for cuMotion
│   └── cumotion/               # cuMotion config yamls
└── docker-compose.yml
```

### 2.2 Containerization

**Server Container (x86 Development):**
```dockerfile
# docker/Dockerfile.server.x86
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    software-properties-common curl gnupg lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-foxglove-bridge \
    ros-humble-moveit \
    python3-colcon-common-extensions

# Install ur_rtde
RUN add-apt-repository ppa:sdurobotics/ur-rtde && \
    apt-get update && apt-get install -y librtde librtde-dev

# Install Isaac ROS packages (cuMotion, nvblox) - requires Isaac ROS workspace setup
# See: https://nvidia-isaac-ros.github.io/getting_started/index.html

WORKDIR /ros2_ws
COPY . /ros2_ws/src/simforge
RUN . /opt/ros/humble/setup.sh && colcon build
```

**Jetson Thor Container (Production):**
```dockerfile
# docker/Dockerfile.server.l4t
# Base image for Jetson with JetPack 7.0 / Isaac ROS 4.0
FROM nvcr.io/nvidia/isaac/ros:humble-aarch64

# ur_rtde must be built from source on ARM64
RUN git clone https://gitlab.com/sdurobotics/ur_rtde.git && \
    cd ur_rtde && \
    git submodule update --init --recursive && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && make install

# Isaac ROS cuMotion and nvblox are pre-installed in base image
```

**⚠️ macOS Client: Do NOT use Docker for ROS 2**
```
Native ROS 2 on macOS is problematic (no official Apple Silicon support).
Instead:
1. Run simforge_client as native Python with websocket-client
2. Use Foxglove Studio (native macOS app) for visualization
3. Connect to Foxglove Bridge on server via WebSocket port 9090
```

### 2.3 Network Configuration

**ROS 2 DDS Configuration (Server-side only):**
```bash
# /etc/ros2/fastdds.xml
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/etc/ros2/fastdds.xml
```

**Foxglove Bridge Launch (Server):**
```python
# launch/foxglove_bridge.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            parameters=[{
                'port': 9090,
                'address': '0.0.0.0',  # Allow external connections
                'send_buffer_limit': 100000000,  # 100MB for point clouds
                'use_compression': True,
            }]
        )
    ])
```

**macOS Client Connection:**
```python
# simforge_client/foxglove_client.py
import asyncio
from foxglove_websocket import Client

async def connect_to_server(server_ip: str):
    async with Client(f"ws://{server_ip}:9090") as client:
        # Subscribe to robot state
        await client.subscribe("/joint_states")
        await client.subscribe("/nvblox/mesh")
        
        # Receive messages
        async for message in client:
            handle_message(message)
```

**Verification Test:**
```bash
# On Server (Workstation)
ros2 launch foxglove_bridge foxglove_bridge.launch.py

# On Mac - open Foxglove Studio app
# Connect to: ws://<WORKSTATION_IP>:9090
# Verify you see ROS 2 topics
---

## 3. Phase 2: Remote Control Refactor (Weeks 3-4)

**Goal:** Decouple current simforge logic so commands originate from Mac but execution happens on Server.

### 3.1 Define Interfaces (ROS 2 Actions)

**MoveRobot.action** (Enhanced with multi-robot support):
```yaml
# msgs/simforge_msgs/action/MoveRobot.action

# === GOAL ===
string robot_name                          # e.g., "ur20_1", "ur5e_2" (for multi-robot)
string reference_frame                     # e.g., "world", "robot_base", "tool0"
uint8 motion_type                          # 0=JOINT, 1=CARTESIAN, 2=TRAJECTORY
geometry_msgs/Pose target_pose             # For Cartesian motion
float64[] target_joints                    # For joint motion [6 values for 6-DOF]
trajectory_msgs/JointTrajectory trajectory # For pre-planned trajectory
float64 velocity_scale                     # 0.0-1.0, default 0.5
float64 acceleration_scale                 # 0.0-1.0, default 0.5
string text_prompt                         # Optional: for VLA-guided motion
bool collision_check_enabled               # Enable cuMotion collision checking
---
# === RESULT ===
bool success
string message
float64[] final_joint_positions            # Actual joint positions at completion
geometry_msgs/Pose final_pose              # Actual TCP pose at completion
float64 execution_time_sec
---
# === FEEDBACK ===
float32 progress                           # 0.0 to 1.0
float64[] current_joint_positions
geometry_msgs/Pose current_pose
string status                              # "planning", "executing", "collision_detected"
```

**RobotState.msg** (Published at 50Hz by server):
```yaml
# msgs/simforge_msgs/msg/RobotState.msg
std_msgs/Header header
string robot_name
float64[] joint_positions                  # [6]
float64[] joint_velocities                 # [6]
float64[] joint_torques                    # [6]
geometry_msgs/Pose tcp_pose
geometry_msgs/Wrench tcp_wrench            # Force/torque at TCP
uint8 robot_mode                           # 0=DISCONNECTED, 1=IDLE, 2=RUNNING, 3=ERROR
bool protective_stop_active
bool emergency_stop_active
```

### 3.2 Refactor simforge_client (Mac)

**Key Change:** Client does NOT run ROS 2. It uses pure WebSocket.

```python
# simforge_client/command_client.py
import asyncio
import json
from websockets import connect

class SimforgeClient:
    """WebSocket client for controlling robots from macOS"""
    
    def __init__(self, server_ip: str, command_port: int = 8765):
        self.server_uri = f"ws://{server_ip}:{command_port}"
        self.ws = None
        self._heartbeat_task = None
        
    async def connect(self):
        self.ws = await connect(self.server_uri)
        self._heartbeat_task = asyncio.create_task(self._send_heartbeat())
        
    async def _send_heartbeat(self):
        """Send heartbeat at 50Hz to maintain safety watchdog"""
        while True:
            await self.ws.send(json.dumps({
                "type": "heartbeat",
                "timestamp_ns": time.time_ns(),
            }))
            await asyncio.sleep(0.02)  # 50Hz
    
    async def move_robot(
        self,
        robot_name: str,
        target_pose: dict = None,
        target_joints: list = None,
        velocity_scale: float = 0.5,
        collision_check: bool = True,
    ) -> dict:
        """Send move command and wait for result"""
        request = {
            "type": "move_robot",
            "robot_name": robot_name,
            "target_pose": target_pose,
            "target_joints": target_joints,
            "velocity_scale": velocity_scale,
            "collision_check_enabled": collision_check,
        }
        await self.ws.send(json.dumps(request))
        
        # Wait for result (with progress updates)
        async for message in self.ws:
            msg = json.loads(message)
            if msg["type"] == "feedback":
                print(f"Progress: {msg['progress']*100:.1f}%")
            elif msg["type"] == "result":
                return msg
    
    async def emergency_stop(self):
        """Trigger immediate stop on all robots"""
        await self.ws.send(json.dumps({"type": "emergency_stop"}))
```

### 3.3 Refactor simforge_server (Workstation)

**Command Gateway Node** - bridges WebSocket commands to ROS 2 Actions:

```python
# simforge_server/nodes/command_gateway_node.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import asyncio
import websockets
import json

from simforge_msgs.action import MoveRobot
from simforge_msgs.msg import Heartbeat

class CommandGatewayNode(Node):
    """Bridge between WebSocket (Mac) and ROS 2 Action Servers"""
    
    def __init__(self):
        super().__init__('command_gateway')
        
        # Action clients for each robot
        self.move_clients = {}  # robot_name -> ActionClient
        
        # Heartbeat publisher (for safety watchdog)
        self.heartbeat_pub = self.create_publisher(Heartbeat, '/safety/heartbeat', 10)
        
        # Start WebSocket server
        self.ws_server = None
        
    def register_robot(self, robot_name: str):
        """Register action client for a robot"""
        self.move_clients[robot_name] = ActionClient(
            self, MoveRobot, f'/{robot_name}/move_robot'
        )
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket messages from Mac client"""
        async for message in websocket:
            msg = json.loads(message)
            
            if msg["type"] == "heartbeat":
                # Forward to ROS 2 for safety watchdog
                hb = Heartbeat()
                hb.timestamp_ns = msg["timestamp_ns"]
                self.heartbeat_pub.publish(hb)
                
            elif msg["type"] == "move_robot":
                robot = msg["robot_name"]
                result = await self.execute_move(robot, msg, websocket)
                await websocket.send(json.dumps({"type": "result", **result}))
                
            elif msg["type"] == "emergency_stop":
                await self.trigger_emergency_stop()
    
    async def execute_move(self, robot_name: str, params: dict, ws) -> dict:
        """Execute move via ROS 2 action with feedback streaming"""
        client = self.move_clients.get(robot_name)
        if not client:
            return {"success": False, "message": f"Unknown robot: {robot_name}"}
        
        goal = MoveRobot.Goal()
        goal.robot_name = robot_name
        goal.velocity_scale = params.get("velocity_scale", 0.5)
        goal.collision_check_enabled = params.get("collision_check_enabled", True)
        
        if params.get("target_joints"):
            goal.motion_type = 0  # JOINT
            goal.target_joints = params["target_joints"]
        elif params.get("target_pose"):
            goal.motion_type = 1  # CARTESIAN
            # Convert dict to Pose msg...
        
        # Send goal with feedback callback
        future = client.send_goal_async(goal, feedback_callback=lambda fb: 
            asyncio.create_task(ws.send(json.dumps({
                "type": "feedback",
                "progress": fb.feedback.progress,
                "status": fb.feedback.status,
            })))
        )
        
        result = await future.result()
        return {
            "success": result.result.success,
            "message": result.result.message,
            "final_joints": list(result.result.final_joint_positions),
        }
```

**Robot Control Node** - wraps ur_ros_rtde:

```python
# simforge_server/nodes/robot_control_node.py
"""
Uses ur_ros_rtde for actual robot control.
Repository: https://github.com/SuperDiodo/ur_ros_rtde

ur_ros_rtde provides:
- RobotStateReceiver: publishes /joint_states, forces, torques
- CommandServer: exposes MoveJ, MoveL, ServoJ as ROS 2 actions
- Easy MoveIt 2 integration
"""

from ur_ros_rtde_msgs.action import MoveJ, MoveL, ServoJ
from ur_ros_rtde_msgs.srv import GetRobotState

class URRobotControlNode(Node):
    def __init__(self, robot_name: str, robot_ip: str):
        super().__init__(f'{robot_name}_control')
        
        # ur_ros_rtde action clients
        self.movej_client = ActionClient(self, MoveJ, f'/{robot_name}/movej')
        self.movel_client = ActionClient(self, MoveL, f'/{robot_name}/movel')
        self.servoj_client = ActionClient(self, ServoJ, f'/{robot_name}/servoj')
        
        # State service
        self.state_client = self.create_client(
            GetRobotState, f'/{robot_name}/get_robot_state'
        )
        
    async def move_joints(self, joints: list, velocity: float, acceleration: float):
        """Execute joint move via ur_ros_rtde"""
        goal = MoveJ.Goal()
        goal.target_joint_positions = joints
        goal.speed = velocity
        goal.acceleration = acceleration
        
        result = await self.movej_client.send_goal_async(goal)
        return result.result
```

### 3.4 Integration with Foxglove Studio (Visualization)

**Topics to visualize in Foxglove:**
| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/joint_states` | sensor_msgs/JointState | Robot joint positions |
| `/robot_state` | simforge_msgs/RobotState | Full robot state |
| `/nvblox/mesh` | nvblox_msgs/Mesh3D | 3D environment mesh |
| `/camera/color/image_raw` | sensor_msgs/Image | RGB camera feed |
| `/camera/depth/image_rect_raw` | sensor_msgs/Image | Depth image |
| `/cumotion/planned_path` | nav_msgs/Path | Planned trajectory |
---

## 4. Phase 3: Perception & Dynamic Avoidance (Weeks 5-7)

> ⚠️ **Timeline Adjusted:** This phase requires 3 weeks due to world model format conversion and cuMotion XRDF setup complexity.

**Goal:** Integrate RealSense depth perception with nvblox for real-time collision-aware motion planning.

### 4.1 Perception Stack Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERCEPTION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  RealSense D435i                                                     │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐                 │
│  │ realsense2_camera   │───►│ isaac_ros_nvblox    │                 │
│  │ /camera/depth/...   │    │                     │                 │
│  │ /camera/color/...   │    │ - TSDF (5cm voxels) │                 │
│  │ /camera/aligned_... │    │ - ESDF (collision)  │                 │
│  └─────────────────────┘    │ - Mesh (vis)        │                 │
│                             └──────────┬──────────┘                 │
│                                        │                             │
│                                        ▼                             │
│                             ┌─────────────────────┐                 │
│                             │ cuMotion Planner    │                 │
│                             │ (nvblox_costmap)    │                 │
│                             │                     │                 │
│                             │ Collision-free      │                 │
│                             │ trajectory output   │                 │
│                             └─────────────────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 RealSense D435i ROS 2 Integration

**Launch Configuration:**
```python
# launch/realsense.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            parameters=[{
                'enable_color': True,
                'enable_depth': True,
                'enable_infra1': False,
                'enable_infra2': False,
                'depth_module.profile': '640x480x30',  # Resolution x FPS
                'rgb_camera.profile': '640x480x30',
                'align_depth.enable': True,            # Critical for nvblox
                'pointcloud.enable': False,            # nvblox handles this
                'initial_reset': True,
            }],
            remappings=[
                ('/camera/camera/depth/image_rect_raw', '/camera/depth/image'),
                ('/camera/camera/color/image_raw', '/camera/color/image'),
                ('/camera/camera/color/camera_info', '/camera/color/camera_info'),
            ]
        )
    ])
```

### 4.3 nvblox Configuration

**Research Findings - nvblox Performance on Jetson Thor:**
| Component | Jetson Orin | **Jetson Thor** | Note |
|-----------|-------------|-----------------|------|
| TSDF | 0.8 ms | **0.4 ms** | 2x faster |
| ESDF | 1.7 ms | **1.0 ms** | Used by cuMotion |
| Meshing | 2.3 ms | **1.4 ms** | For visualization |
| Dynamics | 2.0 ms | **1.4 ms** | Moving obstacle detection |

**nvblox Node Configuration:**
```python
# launch/nvblox.launch.py
Node(
    package='isaac_ros_nvblox',
    executable='nvblox_node',
    parameters=[{
        'voxel_size': 0.05,                    # 5cm voxels (balance accuracy/speed)
        'esdf': True,                          # Enable ESDF for collision checking
        'esdf_2d': False,                      # 3D ESDF for arm manipulation
        'esdf_update_rate_hz': 10.0,           # ESDF update frequency
        'mesh': True,                          # Enable mesh for visualization
        'mesh_update_rate_hz': 5.0,            # Mesh update (can be slower)
        'global_frame': 'world',
        
        # Dynamic object handling
        'use_depth_map': True,
        'use_lidar': False,
        
        # Memory optimization for Jetson
        'max_tsdf_update_blocks': 10000,
        'max_mesh_update_blocks': 5000,
    }],
    remappings=[
        ('depth/image', '/camera/depth/image'),
        ('depth/camera_info', '/camera/depth/camera_info'),
        ('color/image', '/camera/color/image'),
        ('color/camera_info', '/camera/color/camera_info'),
    ]
)
```

### 4.4 cuMotion Integration with MoveIt 2

**Prerequisites:**
1. URDF file for your robot (already have in `assets/`)
2. **XRDF file** - cuMotion's extended robot description (collision spheres, self-collision pairs)
3. MoveIt 2 config package

**XRDF Generation for UR20:**
```bash
# Use NVIDIA's tool to generate XRDF from URDF
# This defines collision spheres for each link
python3 -m curobo.util.make_robot_config \
    --urdf assets/ur20/ur20.urdf \
    --output config/robots/ur20.xrdf
```

**cuMotion MoveIt Plugin Configuration:**
```yaml
# config/cumotion/cumotion_config.yaml
planner_plugin: isaac_ros_cumotion/CumotionMoveItPlugin

isaac_ros_cumotion:
  robot_file: "ur20.xrdf"
  urdf_path: "/ros2_ws/src/simforge/assets/ur20/ur20.urdf"
  
  # Planning parameters
  time_dilation_factor: 0.5          # Conservative speed
  enable_graph_planner: True         # Global planning
  enable_trajopt: True               # Local optimization
  
  # Collision parameters
  collision_check_voxel_size: 0.02   # 2cm collision checking
  collision_cost_weight: 100.0       # High penalty for collisions
  
  # nvblox integration
  use_nvblox_costmap: True
  nvblox_costmap_topic: "/nvblox/esdf"
  costmap_update_timeout: 0.1        # 100ms max wait for costmap
```

**cuMotion Wrapper Node:**
```python
# simforge_server/wrappers/cumotion_wrapper.py
"""
Wrapper for cuMotion that:
1. Receives target pose from MoveRobot action
2. Queries nvblox ESDF for current obstacles
3. Plans collision-free trajectory
4. Returns trajectory to robot control node
"""

from isaac_ros_cumotion_interfaces.srv import PlanTrajectory
from nvblox_msgs.msg import DistanceMapSlice

class CuMotionWrapper(Node):
    def __init__(self):
        super().__init__('cumotion_wrapper')
        
        # cuMotion service client
        self.plan_client = self.create_client(
            PlanTrajectory, '/cumotion/plan_trajectory'
        )
        
        # Subscribe to nvblox ESDF
        self.esdf_sub = self.create_subscription(
            DistanceMapSlice, '/nvblox/esdf', self.esdf_callback, 10
        )
        self.current_esdf = None
        
    async def plan_motion(
        self,
        start_joints: list,
        target_pose: Pose,
        robot_name: str,
    ) -> JointTrajectory:
        """Plan collision-free trajectory using cuMotion + nvblox ESDF"""
        
        request = PlanTrajectory.Request()
        request.start_joint_positions = start_joints
        request.goal_pose = target_pose
        request.robot_config = robot_name
        
        # cuMotion internally uses nvblox ESDF for collision checking
        result = await self.plan_client.call_async(request)
        
        if result.success:
            return result.trajectory
        else:
            raise PlanningFailedException(result.error_message)
```

### 4.5 Dynamic Obstacle Avoidance

> ⚠️ **Research Finding:** True real-time dynamic obstacle avoidance (replanning mid-trajectory when new obstacles appear) is NOT natively supported in cuMotion as of Isaac ROS 4.0. This is tracked as [GitHub Issue #45](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion/issues/45).

**Current Workaround - Stop-Replan Pattern:**
```python
class DynamicAvoidanceNode(Node):
    """
    Monitor ESDF for obstacles entering the planned path.
    If collision imminent: stop robot, replan, resume.
    """
    
    def __init__(self):
        self.current_trajectory = None
        self.trajectory_start_time = None
        
        # 20Hz monitoring
        self.create_timer(0.05, self.check_trajectory_safety)
    
    def check_trajectory_safety(self):
        if self.current_trajectory is None:
            return
            
        # Get points along remaining trajectory
        remaining_points = self.get_remaining_trajectory_points()
        
        # Check each point against current ESDF
        for point in remaining_points:
            distance = self.query_esdf_at_point(point)
            if distance < self.safety_margin:  # 5cm margin
                self.get_logger().warn("Obstacle detected in path - stopping")
                self.trigger_stop_and_replan()
                break
    
    def trigger_stop_and_replan(self):
        # 1. Command immediate stop (deceleration)
        self.robot_stop_client.call_async(StopRequest(deceleration=2.0))
        
        # 2. Wait for stop confirmation
        # 3. Get current position
        # 4. Replan from current to original goal
        # 5. Resume execution
```

**Future: Full Dynamic Avoidance (When Available)**
```python
# This API may be available in Isaac ROS 4.1+
# Concept: cuMotion continuously monitors ESDF and modifies trajectory in real-time
cumotion_config:
  dynamic_replanning: True
  replan_frequency_hz: 10.0
  lookahead_time_sec: 1.0
```

### 4.6 Simulation Parity (Genesis Mock Camera)

For development without physical hardware:
```python
# simforge_server/nodes/mock_camera_node.py
"""
Generate synthetic depth images from Genesis simulation
for testing nvblox + cuMotion pipeline without RealSense
"""

class GenesisMockCameraNode(Node):
    def __init__(self, genesis_client):
        self.genesis = genesis_client
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image', 10)
        self.color_pub = self.create_publisher(Image, '/camera/color/image', 10)
        
        # 30Hz publish rate
        self.create_timer(1/30, self.publish_images)
    
    def publish_images(self):
        # Get depth from Genesis ray-casting
        depth = self.genesis.render_depth(camera_pose=self.camera_pose)
        color = self.genesis.render_rgb(camera_pose=self.camera_pose)
        
        self.depth_pub.publish(self.numpy_to_depth_msg(depth))
        self.color_pub.publish(self.numpy_to_rgb_msg(color))
```
---

## 5. Phase 4: VLA Integration (Weeks 8-10)

> ⚠️ **Timeline Adjusted:** VLA integration requires 3 weeks due to model optimization, quantization, and latency tuning.

**Goal:** Deploy Vision-Language-Action models for autonomous manipulation with human-like instruction following.

### 5.1 VLA Architecture Selection

**Research Finding - VLA Execution Frequency Problem:**
Current VLA models (OpenVLA, RT-2) run at **3-10 Hz** due to large model inference latency. This is too slow for reactive manipulation. The solution is a **Dual-System Architecture**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     DUAL-SYSTEM VLA ARCHITECTURE                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  System 2: VLA Reasoning (Slow - 5Hz)                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │   Image + Text ──► OpenVLA (7B) ──► High-level Action Plan      │   │
│  │                     TensorRT INT4                                │   │
│  │                                                                  │   │
│  │   "Pick up the red cup" ──► [approach, grasp, lift] sequence    │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼ Waypoints/Subgoals (5Hz)                │
│  System 1: Reactive Control (Fast - 100Hz+)                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │   Subgoal + Current State ──► cuMotion ──► Trajectory           │   │
│  │                                                                  │   │
│  │   Real-time collision avoidance, smooth execution               │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 5.2 OpenVLA Deployment on Jetson Thor

**Model Specifications:**
| Property | OpenVLA 7B | OpenVLA 7B (INT4-AWQ) |
|----------|------------|----------------------|
| Parameters | 7B | 7B |
| Memory | ~14GB FP16 | ~4GB INT4 |
| Latency (A100) | ~200ms | ~80ms |
| Latency (Thor, projected) | ~150ms | ~60ms |
| Accuracy Loss | - | ~0.85-1% |

**TensorRT Conversion for Jetson:**
```bash
# Use rail-berkeley/tensorrt-openvla for conversion
# https://github.com/rail-berkeley/tensorrt-openvla

git clone https://github.com/rail-berkeley/tensorrt-openvla
cd tensorrt-openvla

# Convert to TensorRT with INT4-AWQ quantization
python convert_to_tensorrt.py \
    --model openvla/openvla-7b \
    --precision int4 \
    --quantization awq \
    --output /models/openvla_int4_tensorrt \
    --max_batch_size 1

# For Jetson Thor cross-compilation (if building on x86)
# Note: Cross-compilation has known challenges - prefer native build on device
```

### 5.3 VLA Inference Server

**ROS 2 Service Definition:**
```yaml
# msgs/simforge_msgs/srv/GetVLAAction.srv
# Request
sensor_msgs/Image image              # Current camera view
string instruction                   # Natural language command
string[] object_classes              # Optional: objects to attend to
---
# Response
bool success
string error_message
geometry_msgs/Pose[] waypoint_poses  # Sequence of target poses
string[] action_labels               # ["approach", "grasp", "lift"]
float32[] confidence_scores          # Per-waypoint confidence
float32 inference_time_ms
```

**VLA Inference Node:**
```python
# simforge_server/nodes/vla_inference_node.py
import tensorrt as trt
import numpy as np
from transformers import AutoProcessor

class VLAInferenceNode(Node):
    """
    OpenVLA inference server optimized for Jetson Thor.
    Uses TensorRT INT4 for maximum throughput.
    """
    
    def __init__(self):
        super().__init__('vla_inference')
        
        # Load TensorRT engine
        self.engine = self.load_tensorrt_engine('/models/openvla_int4_tensorrt')
        self.context = self.engine.create_execution_context()
        
        # Processor for tokenization
        self.processor = AutoProcessor.from_pretrained('openvla/openvla-7b')
        
        # Service server
        self.srv = self.create_service(
            GetVLAAction, 'vla/get_action', self.handle_request
        )
        
        # Performance monitoring
        self.inference_times = []
        
    def handle_request(self, request, response):
        start = time.time()
        
        # Preprocess image
        image = self.ros_image_to_numpy(request.image)
        
        # Tokenize instruction
        inputs = self.processor(
            images=image,
            text=request.instruction,
            return_tensors='np'
        )
        
        # Run TensorRT inference
        outputs = self.run_inference(inputs)
        
        # Decode action tokens to poses
        waypoints = self.decode_actions(outputs)
        
        response.success = True
        response.waypoint_poses = waypoints
        response.inference_time_ms = (time.time() - start) * 1000
        
        self.get_logger().info(f"VLA inference: {response.inference_time_ms:.1f}ms")
        return response
    
    def run_inference(self, inputs):
        """Execute TensorRT inference"""
        # Allocate device memory
        # Copy inputs to GPU
        # Execute engine
        # Copy outputs from GPU
        # ... TensorRT boilerplate ...
        pass
```

### 5.4 Orchestrator Node (Agent Logic)

**The "Brain" that connects everything:**
```python
# simforge_server/nodes/orchestrator_node.py

class OrchestratorNode(Node):
    """
    High-level agent that orchestrates VLA, perception, and motion.
    
    Flow:
    1. Receive text command from Mac
    2. Get current image from camera
    3. Query VLA for action plan
    4. For each waypoint:
       a. Query cuMotion for collision-free trajectory
       b. Execute trajectory
       c. Check success
    5. Report result to Mac
    """
    
    def __init__(self):
        super().__init__('orchestrator')
        
        # Clients
        self.vla_client = self.create_client(GetVLAAction, 'vla/get_action')
        self.move_client = ActionClient(self, MoveRobot, '/ur20/move_robot')
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image', self.image_callback, 10
        )
        self.current_image = None
        
        # Action server for Mac commands
        self.cmd_server = ActionServer(
            self, ExecuteTask, 'execute_task', self.execute_task
        )
    
    async def execute_task(self, goal_handle):
        """Execute a VLA-guided manipulation task"""
        instruction = goal_handle.request.instruction
        
        # 1. Get VLA action plan
        vla_request = GetVLAAction.Request()
        vla_request.image = self.current_image
        vla_request.instruction = instruction
        
        vla_response = await self.vla_client.call_async(vla_request)
        
        if not vla_response.success:
            goal_handle.abort()
            return ExecuteTask.Result(success=False, message=vla_response.error_message)
        
        # 2. Execute each waypoint
        total_waypoints = len(vla_response.waypoint_poses)
        for i, (pose, label) in enumerate(zip(
            vla_response.waypoint_poses, 
            vla_response.action_labels
        )):
            self.get_logger().info(f"Executing {label} ({i+1}/{total_waypoints})")
            
            # Send to cuMotion via MoveRobot action
            move_goal = MoveRobot.Goal()
            move_goal.target_pose = pose
            move_goal.motion_type = 1  # CARTESIAN
            move_goal.collision_check_enabled = True
            
            result = await self.move_client.send_goal_async(move_goal)
            
            if not result.result.success:
                goal_handle.abort()
                return ExecuteTask.Result(
                    success=False, 
                    message=f"Motion failed at step {label}: {result.result.message}"
                )
            
            # Feedback
            goal_handle.publish_feedback(ExecuteTask.Feedback(
                progress=float(i+1) / total_waypoints,
                current_step=label
            ))
        
        goal_handle.succeed()
        return ExecuteTask.Result(success=True, message="Task completed")
```

### 5.5 Alternative VLA Models (Future Exploration)

| Model | Size | Latency | Notes |
|-------|------|---------|-------|
| OpenVLA | 7B | ~60-150ms | Good baseline, open weights |
| RT-2-X | 55B | ~500ms | Higher accuracy, too slow for real-time |
| Octo | 93M | ~20ms | Very fast, lower generalization |
| Fast-in-Slow (FiS) | 7B+1B | ~20ms (System 1) | Best of both worlds, NeurIPS 2024 |
| π₀ (Physical Intelligence) | 3B | ~30ms | State-of-art, limited availability |

**Recommendation:** Start with OpenVLA INT4. Evaluate Fast-in-Slow when available open-source.
---

## 6. Deployment Strategy

### 6.1 Development → Production Migration

| Environment | Hardware | Purpose |
|-------------|----------|---------|
| **Dev (Current)** | Mac Studio + AI Workstation (x86 GPU) | Rapid iteration, simulation |
| **Staging** | Mac Studio + Jetson Orin AGX | Real hardware validation |
| **Production** | Mac Studio + Jetson Thor | Full performance, VLA at scale |

### 6.2 Jetson Thor Deployment Checklist

When the Jetson Thor arrives:

**Hardware Setup:**
- [ ] Flash JetPack 7.0 (includes Isaac ROS 4.0 support)
- [ ] Connect RealSense D435i via USB 3.0
- [ ] Network configuration (static IP on robot network)
- [ ] Power budget configuration (target 80-100W for full performance)

**Software Deployment:**
```bash
# 1. Transfer Docker image (or rebuild on device)
docker save simforge_server:latest | ssh jetson "docker load"

# 2. Run with NVIDIA runtime for GPU access
docker run -d \
    --runtime nvidia \
    --gpus all \
    --network host \
    -v /dev:/dev \
    --privileged \
    simforge_server:latest

# 3. Verify Isaac ROS packages
ros2 pkg list | grep isaac
# Expected: isaac_ros_nvblox, isaac_ros_cumotion, etc.
```

**Performance Validation:**
```bash
# Test nvblox throughput
ros2 topic hz /nvblox/esdf
# Expected: 10Hz+

# Test cuMotion planning time
ros2 service call /cumotion/benchmark ...
# Expected: <100ms for typical motions

# Test VLA inference
ros2 service call /vla/get_action ...
# Expected: <150ms with INT4 quantization
```

### 6.3 macOS Client - Zero Changes Required

The beauty of this architecture: **the Mac client is unchanged**.

```python
# Same code works for both Workstation and Thor
client = SimforgeClient(server_ip="192.168.1.100")  # Just change IP
await client.connect()
await client.move_robot("ur20", target_pose={...})
```

---

## 7. Revised Timeline Summary

| Phase | Description | Duration | Weeks |
|-------|-------------|----------|-------|
| **0** | Safety Architecture (Heartbeat, Watchdog) | 3 days | Week 1 |
| **1** | Infrastructure (Docker, Foxglove Bridge, Network) | 1.5 weeks | Weeks 1-2 |
| **2** | Remote Control (WebSocket API, ur_ros_rtde, Actions) | 2 weeks | Weeks 3-4 |
| **3** | Perception (RealSense, nvblox, cuMotion) | **3 weeks** | Weeks 5-7 |
| **4** | VLA (OpenVLA TensorRT, Orchestrator) | **3 weeks** | Weeks 8-10 |
| **5** | Integration Testing & Hardening | 1 week | Week 11 |

**Total: ~11 weeks** (vs original 8 weeks)

---

## 8. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| cuMotion dynamic avoidance not available | High | Medium | Implement stop-replan pattern; monitor GitHub #45 |
| VLA latency too high for reactive tasks | Medium | High | Use dual-system architecture; pre-plan approach phases |
| ur_ros_rtde ARM64 build issues | Medium | Medium | Test on Jetson Orin first; maintain build scripts |
| Network latency spikes | Medium | Medium | Buffer commands; use TCP keepalive; monitor with Foxglove |
| RealSense depth noise | Medium | Low | Temporal filtering in nvblox; increase voxel size |
| Thor JetPack 7.0 compatibility | Low | High | Use Isaac ROS 4.0 containers; avoid custom CUDA |

---

## 9. Key Dependencies & Versions

```yaml
# Pinned versions for reproducibility
ros2_distro: humble  # LTS until May 2027
jetpack: "7.0"       # For Jetson Thor
isaac_ros: "4.0"     # Includes cuMotion, nvblox
foxglove_bridge: "0.8.2"
ur_rtde: "1.5.8"     # From sdurobotics PPA
openvla: "0.1.0"     # rail-berkeley
tensorrt: "10.x"     # Blackwell-compatible
cuda: "12.4"         # JetPack 7.0 default
```

---

## 10. Quick Start Commands

**Phase 1 - Test Connectivity:**
```bash
# Server (Workstation)
docker compose up -d ros2_server
ros2 launch foxglove_bridge foxglove_bridge.launch.py

# Mac - Open Foxglove Studio
# Connect to: ws://<SERVER_IP>:9090
```

**Phase 2 - Test Motion:**
```bash
# Server - Start robot control stack
ros2 launch simforge_server robot_bringup.launch.py robot_ip:=192.168.1.9

# Mac - Send test command
python3 simforge_client/test_move.py --server <SERVER_IP> --joints 0 -1.57 1.57 0 0 0
```

**Phase 3 - Test Perception:**
```bash
# Server - Start nvblox with RealSense
ros2 launch simforge_server perception.launch.py

# Foxglove - Visualize /nvblox/mesh topic
```

**Phase 4 - Test VLA:**
```bash
# Server - Start VLA inference
ros2 launch simforge_server vla.launch.py model:=/models/openvla_int4_tensorrt

# Mac - Send instruction
python3 simforge_client/test_vla.py --instruction "pick up the red cube"
```

---

## Appendix A: Message Definitions

<details>
<summary>Click to expand full .msg/.srv/.action definitions</summary>

**Heartbeat.msg:**
```yaml
uint64 timestamp_ns
uint32 sequence_number
string client_id
```

**RobotState.msg:**
```yaml
std_msgs/Header header
string robot_name
float64[6] joint_positions
float64[6] joint_velocities
float64[6] joint_torques
geometry_msgs/Pose tcp_pose
geometry_msgs/Wrench tcp_wrench
uint8 robot_mode
bool protective_stop_active
bool emergency_stop_active
```

**MoveRobot.action:** (See Section 3.1)

**GetVLAAction.srv:** (See Section 5.3)

</details>

---

## Appendix B: Useful Links

- [Isaac ROS cuMotion Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/index.html)
- [Isaac ROS nvblox Documentation](https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/index.html)
- [Foxglove Bridge ROS 2](https://docs.foxglove.dev/docs/connecting-to-data/frameworks/ros2/)
- [ur_ros_rtde GitHub](https://github.com/SuperDiodo/ur_ros_rtde)
- [OpenVLA Paper & Code](https://openvla.github.io/)
- [TensorRT-OpenVLA Conversion](https://github.com/rail-berkeley/tensorrt-openvla)
- [Jetson AGX Thor Specs](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)

---

*Document Version: 2.0 - Enhanced with deep research validation*
