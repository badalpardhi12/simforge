# Motion Planning Implementation Summary

## Overview

This implementation adds GPU-accelerated motion planning with NVIDIA cuMotion and nvblox ESDF integration for collision-free trajectory planning. It replaces the previous demonstration-only joint generation with proper spherical coordinate sampling and inverse kinematics.

## Changes Made

### 1. New Files Created

#### `/simforge_server/utils/pose_generation.py`
- **Purpose**: Spherical coordinate pose sampling ported from `simforge_new/control/proto_simulation.py`
- **Key Functions**:
  - `generate_proto_poses()`: Generate camera poses using spherical coordinates
  - `_spherical_to_cartesian()`: Convert pitch/yaw/distance to XYZ offset
  - `_look_at_quaternion()`: Compute orientation pointing at pivot point
  - `transform_pose_to_world()`: Transform pose from target frame to world frame
- **Usage**: Used by command_gateway_node for proto-sim pose generation

#### `/simforge_server/nodes/motion_planner_node.py`
- **Purpose**: ROS 2 node providing IK solving and trajectory planning service
- **Features**:
  - Multi-solver architecture (cuMotion → trac_ik → KDL fallback)
  - Platform detection (Jetson vs x86)
  - Trajectory optimization with velocity/acceleration limits
  - TF2 integration for frame transformations
- **Service**: `/plan_cartesian_motion` (PlanCartesianMotion.srv)

#### `/msgs/simforge_msgs/srv/PlanCartesianMotion.srv`
- **Purpose**: ROS 2 service definition for motion planning requests
- **Request**: Target pose, reference frame, velocity/acceleration limits
- **Response**: Planned trajectory, IK solution, planning metrics

#### `/config/cumotion/ur5e_cumotion.yaml`
- **Purpose**: cuMotion configuration for UR5e robot
- **Features**: nvblox ESDF integration, joint limits, trajectory parameters

#### `/simforge_server/launch/nvblox.launch.py`
- **Purpose**: Launch configuration for nvblox 3D reconstruction
- **Publishes**: `/nvblox/combined_esdf` for collision checking

#### `/simforge_server/launch/cumotion.launch.py`
- **Purpose**: Launch cuMotion motion planner with MoveIt 2 integration

### 2. Modified Files

#### `/simforge_server/nodes/command_gateway_node.py`
- **Added**: Import of pose_generation utilities
- **Added**: TF2 buffer for target object lookups
- **Updated**: `rpc_run_proto_sim()` to use proper pose generation
- **Added**: `use_ik` parameter for IK mode vs demo mode
- **Renamed**: `_compute_proto_pose_joints()` → `_compute_demo_pose_joints()`
- **Added**: `_solve_ik_for_pose()` with analytical IK approximation
- **Added**: Pose position/orientation in feedback messages

#### `/msgs/simforge_msgs/CMakeLists.txt`
- **Added**: `PlanCartesianMotion.srv` to service list

#### `/docker/Dockerfile.server.l4t`
- **Updated**: Base image to Isaac ROS 3.2.0
- **Added**: Isaac ROS cuMotion and nvblox packages
- **Added**: trac_ik as IK fallback
- **Added**: CUDA_HOME environment variable

#### `/docker/Dockerfile.server.x86`
- **Added**: trac_ik-lib and kdl-parser-py packages

#### `/simforge_server/CMakeLists.txt`
- **Added**: motion_planner_node.py to installed executables
- **Added**: utils directory installation

### 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Mac Client (SwiftUI)                      │
│               Proto-Sim Control Interface                    │
└─────────────────────────────────────────────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Command Gateway Node                        │
│  - Receives proto-sim parameters                             │
│  - Generates poses using pose_generation.py                  │
│  - Calls motion planner for IK (optional)                   │
│  - Publishes joint states for visualization                  │
└─────────────────────────────────────────────────────────────┘
                              │ ROS 2 Service
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Motion Planner Node                         │
│  - Receives Cartesian target pose                           │
│  - Transforms pose using TF2                                 │
│  - Solves IK (cuMotion on Jetson, trac_ik on x86)           │
│  - Plans collision-free trajectory                           │
│  - Returns JointTrajectory                                   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│    cuMotion (Jetson)    │     │    trac_ik (x86)        │
│  - GPU-accelerated IK   │     │  - CPU IK solver        │
│  - Trajectory opt       │     │  - KDL fallback         │
│  - nvblox ESDF          │     │                         │
└─────────────────────────┘     └─────────────────────────┘
```

## Usage

### Proto-Sim with Demo Mode (Default)
```python
# Client sends:
{
    "method": "run_proto_sim",
    "params": {
        "horiz": [0],
        "vert": [0],
        "distance": [250, 350, 450, 550],
        "roll": [-90],
        "pitch": [-45, -30, 0, 15],
        "yaw": [-30, 0, 30],
        "use_ik": false  # Demo mode - visually interesting movements
    }
}
```

### Proto-Sim with IK Mode
```python
# Client sends:
{
    "method": "run_proto_sim",
    "params": {
        "target_object": "face_link",  # Must be in TF tree
        "use_ik": true,  # Enable actual IK solving
        "distance": [300, 400],
        "pitch": [-30, 0, 30],
        "yaw": [-45, 0, 45]
    }
}
```

## Building

```bash
# Rebuild messages
cd /ros2_ws
colcon build --packages-select simforge_msgs

# Rebuild server
colcon build --packages-select simforge_server

# Source
source install/setup.bash
```

## Launching

### x86 (Development)
```bash
ros2 launch simforge_server robot_bringup.launch.py
```

### Jetson Thor (Production with cuMotion)
```bash
# Launch nvblox for 3D reconstruction
ros2 launch simforge_server nvblox.launch.py

# Launch cuMotion planner
ros2 launch simforge_server cumotion.launch.py

# Launch robot control
ros2 launch simforge_server robot_bringup.launch.py
```

## Next Steps

1. **XRDF Generation**: Generate UR5e XRDF file for cuMotion (uses cuRobo's robot config format)
2. **Calibration**: Add camera-to-robot calibration for accurate TF
3. **Testing**: Test IK mode with real target object transforms
4. **MoveIt 2**: Complete MoveIt 2 configuration for planning interface
5. **Real Robot**: Enable real robot execution mode
