cmake_minimum_required(VERSION 3.5)
project(valid8_cell_description)

find_package(ament_cmake REQUIRED)

# Install directories
install(
  DIRECTORY launch meshes rviz urdf config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
