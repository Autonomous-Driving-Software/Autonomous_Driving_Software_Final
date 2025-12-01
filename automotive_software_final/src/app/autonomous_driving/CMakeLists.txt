################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.5)
project(autonomous_driving)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################################################################################
# Find and load build settings from external packages
################################################################################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ad_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

## Add include directories
include_directories(
    include
    ${CMAKE_SOURCE_DIR}/../../bsw/system/interface
    ${CMAKE_SOURCE_DIR}/../../bsw/system/bridge
)

## Declare a C++ executable

#1. Perception Node
add_executable(perception_node
    src/perception_node.cpp
)
ament_target_dependencies(perception_node
    rclcpp
    std_msgs
    ad_msgs
    tf2
    tf2_ros
)

#2. Planning Node
add_executable(planning_node
    src/planning_node.cpp
)
ament_target_dependencies(planning_node
    rclcpp
    std_msgs
    ad_msgs
    tf2
    tf2_ros
)

#3. Control Node
add_executable(control_node
    src/control_node.cpp
)
ament_target_dependencies(control_node
    rclcpp
    std_msgs
    ad_msgs
    tf2
    tf2_ros
)

## Install all three executables
install(TARGETS 
    perception_node
    planning_node
    control_node
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/simultaneous_inputs.py
  DESTINATION lib/${PROJECT_NAME}
)

##################################################################
# Find and load build settings from external packages
##################################################################
ament_package()