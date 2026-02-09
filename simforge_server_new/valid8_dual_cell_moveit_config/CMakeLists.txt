cmake_minimum_required(VERSION 3.8)
project(valid8_dual_cell_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  srdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
