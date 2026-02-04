# Motion Planning Implementation Fix

## Date: 2026-02-03

## Issues Identified

### 1. Position Sampling Issues
The current `_compute_demo_pose_joints()` function generates arbitrary "visually dramatic" joint configurations instead of solving IK for actual target poses. This is why poses look incorrect.

**Current Problem Code (command_gateway_node.py:848-895):**
```python
def _compute_demo_pose_joints(self, ...):
    # This generates FAKE joints that just look interesting
    j0 = math.radians(yaw_deg) * 0.5 + horiz_m * 2.0  # NOT REAL IK
    j1 = -math.pi/2 + math.radians(pitch_deg) * 0.5 + vert_m * 1.5  # NOT REAL IK
    # ... all these are arbitrary, NOT IK solutions
```

### 2. No Real IK Solving
The `_solve_ik_for_pose()` function has a "simple analytical approximation" that is incorrect for UR5e. It uses oversimplified 2R planar IK that doesn't account for:
- 6-DOF kinematic chain
- Wrist singularities  
- Multiple IK solutions (elbow up/down, etc.)
- Self-collision avoidance

### 3. No Collision Checking
There is NO collision checking in the current execution path:
- `motion_planner_node.py` was created but never integrated
- FCL collision checker exists but is not called
- MoveIt 2 is not launched or used

### 4. No Path Planning
Movement is done via direct joint interpolation (`_execute_sim_movement`) which:
- Does NOT check for collisions along the path
- Does NOT use time-optimal trajectory generation
- Does NOT validate joint limits properly

### 5. WebSocket Connection Issues
The client logs show "cannot call recv while another coroutine is already waiting" - this is a client-side bug where multiple coroutines try to receive from the same WebSocket.

---

## Architecture Correction: NO DEMO MODE

**Key Principle:** The system should ALWAYS use proper IK solving, collision checking, and path planning - regardless of whether in simulation or real robot mode.

### Platform-Specific Motion Planning Stack

| Platform | IK Solver | Path Planner | Collision Checker |
|----------|-----------|--------------|-------------------|
| **AI Workstation (x86)** | KDL via MoveIt 2 | OMPL (RRT-Connect) | FCL via MoveIt 2 |
| **Jetson Thor (ARM64)** | cuMotion IK | cuMotion + TrajOpt | nvblox ESDF + cuMotion |

Both platforms use the SAME interface (`PlanCartesianMotion.srv`) but different backends.

---

## Implementation Plan

### Step 1: Launch MoveIt 2 for IK + Planning (x86)

**New launch file:** `moveit_planning.launch.py`
```python
# Launch MoveIt 2 move_group node with OMPL planning
# This provides:
# - /compute_ik service (IK solving)
# - /plan_kinematic_path service (path planning)
# - /get_planning_scene service (collision world)
```

### Step 2: Implement MoveIt 2 Client in command_gateway

Replace `_solve_ik_for_pose()` with actual MoveIt 2 service calls:

```python
async def _solve_ik_moveit(self, target_pose: Pose) -> Optional[List[float]]:
    """Call MoveIt 2's compute_ik service."""
    if not self.compute_ik_client.service_is_ready():
        return None
    
    request = GetPositionIK.Request()
    request.ik_request.group_name = "ur_manipulator"
    request.ik_request.pose_stamped.header.frame_id = "base_link"
    request.ik_request.pose_stamped.pose = target_pose
    request.ik_request.robot_state = self.current_robot_state
    
    response = await self.compute_ik_client.call_async(request)
    if response.error_code.val == MoveItErrorCodes.SUCCESS:
        return list(response.solution.joint_state.position)
    return None
```

### Step 3: Implement Path Planning with Collision Checking

Replace `_execute_sim_movement()` with MoveIt 2 path planning:

```python
async def _plan_and_execute(
    self,
    target_joints: List[float],
    current_joints: List[float],
) -> bool:
    """Plan collision-free path using MoveIt 2."""
    
    # Create motion plan request
    request = GetMotionPlan.Request()
    request.motion_plan_request.group_name = "ur_manipulator"
    request.motion_plan_request.start_state.joint_state.position = current_joints
    
    # Set goal
    goal = Constraints()
    for i, name in enumerate(self.joint_names):
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = target_joints[i]
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        goal.joint_constraints.append(jc)
    request.motion_plan_request.goal_constraints.append(goal)
    
    # Plan
    response = await self.plan_client.call_async(request)
    if response.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
        return False
    
    # Execute (publish trajectory)
    trajectory = response.motion_plan_response.trajectory.joint_trajectory
    await self._execute_trajectory(trajectory)
    return True
```

### Step 4: Fix WebSocket Client (Mac side)

The client needs to handle reconnection properly without creating multiple recv() tasks:

```python
class CommandClient:
    def __init__(self):
        self._recv_lock = asyncio.Lock()
        self._websocket = None
    
    async def receive(self):
        async with self._recv_lock:
            return await self._websocket.recv()
```

---

## Required Changes

### 1. simforge_server/launch/full_stack.launch.py
- ADD: Launch MoveIt 2 move_group node
- ADD: Load robot URDF and SRDF
- ADD: Configure OMPL planning pipeline

### 2. simforge_server/nodes/command_gateway_node.py
- REMOVE: `_compute_demo_pose_joints()` - DELETE this function entirely
- REPLACE: `_solve_ik_for_pose()` with MoveIt 2 service client
- REPLACE: `_execute_sim_movement()` with trajectory execution via MoveIt 2
- ADD: Service clients for `/compute_ik`, `/plan_kinematic_path`
- ADD: Proper robot state tracking

### 3. simforge_server/config/moveit/
- CREATE: ur5e_moveit_config/ with:
  - ur5e.srdf (semantic robot description)
  - joint_limits.yaml
  - kinematics.yaml (KDL solver config)
  - ompl_planning.yaml

### 4. simforge_client/command_client.py
- FIX: Add recv lock to prevent concurrent recv() calls
- FIX: Proper reconnection state management

### 5. Dockerfile.server.x86
- ADD: ros-humble-moveit
- ADD: ros-humble-moveit-planners-ompl
- ADD: ros-humble-moveit-ros-move-group

---

## Verification Steps

1. **IK Test:**
   ```bash
   ros2 service call /compute_ik moveit_msgs/srv/GetPositionIK "{...}"
   ```

2. **Planning Test:**
   ```bash
   ros2 service call /plan_kinematic_path moveit_msgs/srv/GetMotionPlan "{...}"
   ```

3. **Collision Test:**
   - Add obstacle to planning scene
   - Request motion that would collide
   - Verify planner finds alternative path or rejects

4. **End-to-End Test:**
   - Start proto-sim from client
   - Verify robot moves to correct poses (not random demo poses)
   - Verify no collisions with face_link or table
