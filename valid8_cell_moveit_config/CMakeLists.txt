cmake_minimum_required(VERSION 3.22)
project(valid8_cell_moveit_config)

find_package(ament_cmake REQUIRED)

ament_package()

# Install config directory
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

# Install launch files (exclude setup_assistant.launch.py from production)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch.py" EXCLUDE)

# Install the .setup_assistant file for regeneration
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})
