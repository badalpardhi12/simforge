cmake_minimum_required(VERSION 3.8)
project(simforge_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Declare message, service, and action files
rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/Heartbeat.msg"
  "msg/RobotState.msg"
  "msg/SafetyStatus.msg"
  
  # Services
  "srv/GetVLAAction.srv"
  "srv/GetRobotState.srv"
  "srv/SetSafetyMode.srv"
  "srv/PlanCartesianMotion.srv"
  "srv/MoveJoints.srv"
  
  # Actions
  "action/MoveRobot.action"
  "action/ExecuteTask.action"
  
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
    trajectory_msgs
    builtin_interfaces
)

ament_package()
