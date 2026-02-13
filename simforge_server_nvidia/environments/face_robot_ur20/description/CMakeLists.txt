cmake_minimum_required(VERSION 3.8)
project(face_robot_ur20_description)

find_package(ament_cmake REQUIRED)

# Install URDF / xacro files
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

# Install meshes
install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
