# Valid8 Robot Cell - ROS 2 Integration

This directory contains the ROS 2 packages for the Valid8 robot cell, following the official 
[Universal Robots ROS 2 Custom Workcell Tutorial](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/doc/ur_tutorials/my_robot_cell/doc/index.html).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Valid8 Robot Cell                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────┐              │
│  │ valid8_cell_description  │    │  valid8_cell_control     │              │
│  │                          │    │                          │              │
│  │ • valid8_cell_macro.xacro│───▶│ • valid8_cell_controlled │              │
│  │ • valid8_cell.urdf.xacro │    │   .urdf.xacro            │              │
│  │ • Meshes (table, face)   │    │ • rsp.launch.py          │              │
│  │ • RViz config            │    │ • start_robot.launch.py  │              │
│  │                          │    │ • full_stack.launch.py   │              │
│  └──────────────────────────┘    │ • Calibration config     │              │
│                                  └──────────────────────────┘              │
│                                            │                               │
│                                            ▼                               │
│  ┌──────────────────────────┐    ┌──────────────────────────┐              │
│  │ valid8_cell_moveit_config│◀───│  ur_robot_driver         │              │
│  │                          │    │  (UR official package)   │              │
│  │ • SRDF                   │    │                          │              │
│  │ • Kinematics config      │    │ Uses description_launchfile              │
│  │ • OMPL planning config   │    │ parameter to load our     │              │
│  │ • Controllers config     │    │ custom workcell description│             │
│  │ • Joint limits           │    │                          │              │
│  └──────────────────────────┘    └──────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Why This Architecture?

The previous implementation had issues:
1. **TF Conflicts**: Our static URDF publisher conflicted with UR driver's robot_state_publisher
2. **Robot at Origin**: UR driver places robot at world origin by default
3. **No Multi-Robot Support**: No tf_prefix for namespacing robot frames

The new architecture solves these by:
1. **Custom Description Launch File**: We pass our `rsp.launch.py` to UR driver via `description_launchfile` parameter
2. **Workcell Macro**: Robot is attached to `robot_mount` link which is properly positioned in the workcell
3. **TF Prefix Support**: All robot frames can be prefixed for multi-robot scenarios

### TF Tree

```
world
  └── shop_floor (floor at z=0)
        ├── table_link (optical table at z=1.0)
        ├── face_link (face fixture at [0.1742, 0, 1.6])
        └── robot_mount (at [-0.6758, 0, 1.03] with -90° yaw)
              └── [tf_prefix]base_link
                    └── [tf_prefix]shoulder_link
                          └── ... (robot kinematic chain)
```

## Package Descriptions

### valid8_cell_description

Contains the URDF/XACRO description of the workcell:

- `urdf/valid8_cell_macro.xacro` - Macro defining optical table, face fixture, and robot mount
- `urdf/valid8_cell.urdf.xacro` - Main description file combining workcell + UR robot
- `meshes/` - Mesh files for optical table and face
- `launch/view_robot.launch.py` - View the cell in RViz with joint sliders
- `rviz/view_robot.rviz` - RViz configuration

### valid8_cell_control

Contains ros2_control enabled description and launch files:

- `urdf/valid8_cell_controlled.urdf.xacro` - Adds ros2_control hardware interface
- `launch/rsp.launch.py` - Robot state publisher (passed to UR driver)
- `launch/start_robot.launch.py` - Start robot with UR driver
- `launch/full_stack.launch.py` - Complete system launch
- `config/kinematics_calibration.yaml` - Robot-specific calibration
- `scripts/extract_calibration.py` - Extract calibration from real robot

### valid8_cell_moveit_config

MoveIt 2 configuration:

- `config/valid8_cell.srdf` - Semantic robot description
- `config/kinematics.yaml` - IK solver configuration
- `config/ompl_planning.yaml` - Motion planner configuration
- `config/joint_limits.yaml` - Joint limits with acceleration
- `config/moveit_controllers.yaml` - Controller configuration
- `launch/move_group.launch.py` - MoveIt move_group launch
- `launch/setup_assistant.launch.py` - Regenerate config with Setup Assistant

## Quick Start

### 1. Build the Workspace

```bash
cd ~/colcon_ws
colcon build --packages-select valid8_cell_description valid8_cell_control valid8_cell_moveit_config
source install/setup.bash
```

### 2. Test Description (Visualization Only)

```bash
ros2 launch valid8_cell_description view_robot.launch.py
```

### 3. Start with Mock Hardware (Simulation)

```bash
ros2 launch valid8_cell_control start_robot.launch.py use_mock_hardware:=true
```

### 4. Start with Real Robot

```bash
# First, extract calibration from your robot (one-time setup)
python3 valid8_cell_control/scripts/extract_calibration.py --robot-ip 192.168.1.9

# Then start the robot
ros2 launch valid8_cell_control start_robot.launch.py robot_ip:=192.168.1.9
```

### 5. Full Stack with MoveIt and Foxglove

```bash
# Mock hardware
ros2 launch valid8_cell_control full_stack.launch.py use_mock_hardware:=true

# Real robot
ros2 launch valid8_cell_control full_stack.launch.py robot_ip:=192.168.1.9
```

## Robot Calibration

For accurate kinematic calculations, extract calibration from your real robot:

```bash
# Using the provided script
python3 valid8_cell_control/scripts/extract_calibration.py --robot-ip 192.168.1.9

# Or using ur_calibration directly
ros2 launch ur_calibration calibration_correction.launch.py \
    robot_ip:=192.168.1.9 \
    target_filename:=valid8_cell_control/config/kinematics_calibration.yaml
```

## Multi-Robot Support

For multi-robot setups, use the `tf_prefix` parameter:

```bash
# Robot 1
ros2 launch valid8_cell_control start_robot.launch.py \
    tf_prefix:=robot1_ \
    robot_ip:=192.168.1.9

# Robot 2 (in another terminal)
ros2 launch valid8_cell_control start_robot.launch.py \
    tf_prefix:=robot2_ \
    robot_ip:=192.168.1.10
```

## Regenerating MoveIt Config

To update the MoveIt configuration (e.g., after URDF changes):

```bash
ros2 launch valid8_cell_moveit_config setup_assistant.launch.py
```

## Troubleshooting

### Robot not visible in Foxglove

1. Check that Foxglove Bridge is running: `ros2 node list | grep foxglove`
2. Verify TF is publishing: `ros2 topic echo /tf`
3. Check robot_description is published: `ros2 topic echo /robot_description`

### IK/Planning failures

1. Ensure MoveIt is launched: `ros2 node list | grep move_group`
2. Check SRDF is loaded correctly in move_group logs
3. Verify acceleration limits are set in `joint_limits.yaml`

### Trajectory execution fails

1. Ensure `scaled_joint_trajectory_controller` is active:
   ```bash
   ros2 control list_controllers
   ```
2. Check robot connection in dashboard client

## References

- [UR ROS 2 Custom Workcell Tutorial](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/doc/ur_tutorials/my_robot_cell/doc/index.html)
- [UR Robot Driver Documentation](https://docs.ros.org/en/ros2_packages/humble/api/ur_robot_driver/)
- [MoveIt 2 Documentation](https://moveit.picknik.ai/main/index.html)
