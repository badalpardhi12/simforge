# Complete robot state message
# Published by server at 50Hz

std_msgs/Header header

# Robot identifier (e.g., "ur20_1", "ur5e_2")
string robot_name

# Joint state (6-DOF for UR robots)
float64[6] joint_positions      # radians
float64[6] joint_velocities     # rad/s
float64[6] joint_torques        # Nm

# TCP (Tool Center Point) state
geometry_msgs/Pose tcp_pose
geometry_msgs/Twist tcp_velocity
geometry_msgs/Wrench tcp_wrench  # Force/torque at TCP

# Robot operational mode
# 0=DISCONNECTED, 1=IDLE, 2=RUNNING, 3=FREEDRIVE, 4=ERROR
uint8 ROBOT_MODE_DISCONNECTED = 0
uint8 ROBOT_MODE_IDLE = 1
uint8 ROBOT_MODE_RUNNING = 2
uint8 ROBOT_MODE_FREEDRIVE = 3
uint8 ROBOT_MODE_ERROR = 4
uint8 robot_mode

# Safety state
bool protective_stop_active
bool emergency_stop_active
bool safety_mode_active

# Connection quality
float32 communication_latency_ms
uint32 packet_loss_count

# Digital I/O state (for gripper, etc.)
bool[8] digital_inputs
bool[8] digital_outputs
