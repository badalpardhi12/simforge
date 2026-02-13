"""
simforge_gateway_nvidia — cuRobo GPU Motion Planning Gateway

WebSocket server with NVIDIA cuRobo GPU-accelerated motion planning
and ur_rtde real-robot control for multi-environment robot cells.

Modules:
    config               Robot configuration, constants, dataclasses
    env_loader           Environment YAML configuration loader
    curobo_planner       cuRobo GPU motion planning (MotionGen / IK)
    collision_matrix     Self-collision matrix generation
    trajectory_executor  Trajectory dispatch (RTDE / ROS2)
    rtde_controller      ur_rtde low-level wrapper
    joint_state_manager  Mode-aware /joint_states publishing
    protocol_executor    Multi-pose protocol runner
    rpc_handlers         WebSocket RPC implementations
"""
