# -*- coding: utf-8 -*-
"""ROS 2 node exposing the ``gpu_planning`` planner zoo (MoveIt-free).

Offers a ``gpu_planning_msgs/PlanJointPath`` service (default name
``~/plan_joint_path``) that plans a joint-space start->goal request with one of
the six substrate-consistent GPU planners, and subscribes to
``gpu_planning_msgs/CollisionScene`` (default ``/collision_scene``) to keep the
shared collision world in sync. Built on ROS 2 common_interfaces only — no
MoveIt dependency.

Parameters (all overridable on the command line / launch):

* ``robot_asset`` (str, default ""): path to a robot YAML asset; empty selects
  the package's bundled Panda.
* ``device`` (str, default "cuda"): torch device for the scene tensors.
* ``default_planner`` (str, default "prrtc"): planner used when a request's
  ``planner_id`` is empty. One of prrtc / mitstar / mhastar / wpase / stomp / chomp.
* ``service_name`` (str, default "~/plan_joint_path").
* ``collision_scene_topic`` (str, default "/collision_scene").

Requires the compiled ``prrtc`` pybind module and a CUDA device at *runtime*;
the module is imported lazily inside the plugins, so this node file imports
without a GPU present (only planning itself needs one).
"""

from __future__ import annotations

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gpu_planning_msgs.msg import CollisionScene
from gpu_planning_msgs.srv import PlanJointPath

from . import conversions

# Make the sibling ``gpu_planning`` package importable from a source checkout
# (ros2/gpu_planning_ros2/gpu_planning_ros2/planner_node.py -> code/gpu_planning/python).
_PKG_PY = Path(__file__).resolve().parents[3] / "python"
if _PKG_PY.is_dir() and str(_PKG_PY) not in sys.path:
    sys.path.insert(0, str(_PKG_PY))


def _planner_registry():
    """Map lowercased planner ids / labels to plugin classes (lazy import)."""
    from gpu_planning import (
        CHOMPPlanner,
        MHAStarPlanner,
        MITStarPlanner,
        PRRTCPlanner,
        STOMPPlanner,
        WPASEPlanner,
    )
    return {
        "prrtc": PRRTCPlanner,
        "mitstar": MITStarPlanner,
        "mit*": MITStarPlanner,
        "mhastar": MHAStarPlanner,
        "mha*": MHAStarPlanner,
        "wpase": WPASEPlanner,
        "wpa*se": WPASEPlanner,
        "stomp": STOMPPlanner,
        "chomp": CHOMPPlanner,
    }


class GpuPlanningNode(Node):
    """rclpy node wrapping the substrate planner plugins."""

    def __init__(self):
        super().__init__("gpu_planning_node")

        self.declare_parameter("robot_asset", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("default_planner", "prrtc")
        self.declare_parameter("service_name", "~/plan_joint_path")
        self.declare_parameter("collision_scene_topic", "/collision_scene")

        self._asset = self.get_parameter("robot_asset").value or None
        self._device = self.get_parameter("device").value
        self._default_planner = (
            self.get_parameter("default_planner").value or "prrtc").lower()

        from gpu_planning import PlanningScene
        self._scene = PlanningScene.from_robot_asset(self._asset, self._device)
        self._registry = _planner_registry()
        self._plugins = {}  # id -> initialised plugin (lazy, cached)

        service_name = self.get_parameter("service_name").value
        scene_topic = self.get_parameter("collision_scene_topic").value
        self._srv = self.create_service(
            PlanJointPath, service_name, self._on_plan_request)
        self._scene_sub = self.create_subscription(
            CollisionScene, scene_topic, self._on_collision_scene, 1)

        self.get_logger().info(
            f"gpu_planning_node ready: service='{service_name}', "
            f"scene_topic='{scene_topic}', default_planner="
            f"'{self._default_planner}', device='{self._device}'")

    # ------------------------------------------------------------------ #
    # Collision-world sync
    # ------------------------------------------------------------------ #
    def _on_collision_scene(self, msg: CollisionScene) -> None:
        try:
            cuboids = conversions.collision_objects_to_cuboids(list(msg.objects))
            self._scene.update_world(cuboids)
            self.get_logger().debug(
                f"collision scene updated: {len(cuboids)} OBB(s)")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(f"failed to apply collision scene: {e}")

    # ------------------------------------------------------------------ #
    # Planner lifecycle
    # ------------------------------------------------------------------ #
    def _get_plugin(self, planner_id: str):
        key = (planner_id or self._default_planner).lower()
        cls = self._registry.get(key)
        if cls is None:
            cls = self._registry[self._default_planner]
            self.get_logger().warning(
                f"unknown planner_id '{planner_id}', "
                f"falling back to '{self._default_planner}'")
            key = self._default_planner
        if key not in self._plugins:
            plugin = cls()
            if not plugin.initialize(self._scene, asset_path=self._asset):
                raise RuntimeError(f"failed to initialize planner '{key}'")
            self._plugins[key] = plugin
        return self._plugins[key]

    # ------------------------------------------------------------------ #
    # Service handler
    # ------------------------------------------------------------------ #
    def _on_plan_request(self, request, response):
        from gpu_planning import PlanningRequest

        try:
            joint_names, start_vec, goal_vec = conversions.extract_start_goal(
                request.start, request.goal)
        except ValueError as e:
            self.get_logger().warning(f"invalid request: {e}")
            response.success = False
            response.status = f"InvalidRequest: {e}"
            return response

        planner_id = request.planner_id or self._default_planner
        try:
            plugin = self._get_plugin(request.planner_id)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"planner init failed: {e}")
            response.success = False
            response.status = f"InitFailed: {e}"
            return response

        plan_req = PlanningRequest(
            robot_id=request.robot_id or self._scene.robot_id,
            start_joint_state=start_vec,
            target_joint_state=goal_vec,
            velocity_scaling=float(request.max_velocity_scaling_factor or 1.0),
            acceleration_scaling=float(
                request.max_acceleration_scaling_factor or 1.0),
            timeout=float(request.allowed_planning_time or 10.0),
        )

        try:
            result = plugin.plan(plan_req)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"planning raised: {e}")
            response.success = False
            response.status = f"PlannerError: {e}"
            return response

        response.planning_time = float(getattr(result, "solve_time", 0.0))
        response.status = result.status
        if not result.success or result.trajectory is None:
            response.success = False
            self.get_logger().info(
                f"plan failed ({result.status}) with '{planner_id}'")
            return response

        response.success = True
        response.trajectory = self._build_joint_trajectory(joint_names, result)
        self.get_logger().info(
            f"planned {result.num_waypoints} pts in "
            f"{response.planning_time * 1e3:.1f} ms with '{planner_id}'")
        return response

    # ------------------------------------------------------------------ #
    def _build_joint_trajectory(self, joint_names, result) -> JointTrajectory:
        jt = JointTrajectory()
        jt.joint_names = list(joint_names)
        for pos, vel, t in conversions.result_points(result):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in pos]
            pt.velocities = [float(x) for x in vel]
            sec = int(t)
            pt.time_from_start = Duration(
                sec=sec, nanosec=int(round((t - sec) * 1e9)))
            jt.points.append(pt)
        return jt


def main(argv=None):
    rclpy.init(args=argv)
    node = GpuPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for plugin in node._plugins.values():
            try:
                plugin.shutdown()
            except Exception:  # noqa: BLE001
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
