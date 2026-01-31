# Safety status message
# Published by safety watchdog node

std_msgs/Header header

# Overall safety state
# 0=NORMAL, 1=WARNING, 2=PROTECTIVE_STOP, 3=EMERGENCY_STOP
uint8 SAFETY_STATE_NORMAL = 0
uint8 SAFETY_STATE_WARNING = 1
uint8 SAFETY_STATE_PROTECTIVE_STOP = 2
uint8 SAFETY_STATE_EMERGENCY_STOP = 3
uint8 safety_state

# Heartbeat status
bool heartbeat_active
uint32 consecutive_heartbeat_misses
float64 last_heartbeat_time
float64 heartbeat_timeout

# Connected clients
string[] connected_client_ids

# Fault information
string[] active_faults
string last_fault_message
builtin_interfaces/Time last_fault_time

# System health
float32 system_cpu_usage
float32 system_memory_usage
float32 gpu_temperature
