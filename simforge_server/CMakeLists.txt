cmake_minimum_required(VERSION 3.8)
project(simforge_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  config
  assets
  DESTINATION share/${PROJECT_NAME}
)

# Install Python executables
install(PROGRAMS
  nodes/safety_watchdog_node.py
  nodes/command_gateway_node.py
  nodes/robot_control_node.py
  nodes/robot_control_node_v2.py
  nodes/ur_communication.py
  nodes/perception_node.py
  nodes/vla_inference_node.py
  nodes/orchestrator_node.py
  nodes/static_transform_republisher.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
