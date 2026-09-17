from setuptools import find_packages, setup

package_name = "gpu_planning_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/gpu_planning.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="wuh15",
    maintainer_email="wuh15@users.noreply.github.com",
    description="ROS2 adapter exposing the gpu_planning substrate planners via "
                "gpu_planning_msgs/PlanJointPath (MoveIt-free).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gpu_planning_node = gpu_planning_ros2.planner_node:main",
        ],
    },
)
