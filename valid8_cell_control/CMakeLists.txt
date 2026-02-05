cmake_minimum_required(VERSION 3.8)
project(valid8_cell_control)

find_package(ament_cmake REQUIRED)

# Install directories
install(
  DIRECTORY launch config urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
