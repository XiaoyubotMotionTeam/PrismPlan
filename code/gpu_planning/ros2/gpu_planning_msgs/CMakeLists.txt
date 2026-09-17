cmake_minimum_required(VERSION 3.8)
project(gpu_planning_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CollisionObject.msg"
  "msg/CollisionScene.msg"
  "srv/PlanJointPath.srv"
  DEPENDENCIES std_msgs geometry_msgs shape_msgs sensor_msgs trajectory_msgs
)

ament_package()
