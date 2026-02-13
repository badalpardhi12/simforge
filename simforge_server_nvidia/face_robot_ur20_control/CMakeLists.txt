cmake_minimum_required(VERSION 3.8)
project(face_robot_ur20_control)

find_package(ament_cmake REQUIRED)

# Install URDF / xacro files
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

# Install controller config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
