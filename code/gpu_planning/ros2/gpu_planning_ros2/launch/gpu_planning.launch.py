# -*- coding: utf-8 -*-
"""Launch the gpu_planning ROS2 node with common parameters."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_asset = LaunchConfiguration("robot_asset")
    device = LaunchConfiguration("device")
    default_planner = LaunchConfiguration("default_planner")

    return LaunchDescription([
        DeclareLaunchArgument("robot_asset", default_value="",
                              description="robot YAML asset (empty = bundled panda)"),
        DeclareLaunchArgument("device", default_value="cuda",
                              description="torch device for scene tensors"),
        DeclareLaunchArgument("default_planner", default_value="prrtc",
                              description="prrtc|mitstar|mhastar|wpase|stomp|chomp"),
        Node(
            package="gpu_planning_ros2",
            executable="gpu_planning_node",
            name="gpu_planning_node",
            output="screen",
            parameters=[{
                "robot_asset": robot_asset,
                "device": device,
                "default_planner": default_planner,
            }],
        ),
    ])
