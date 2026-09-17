# -*- coding: utf-8 -*-
"""Unit tests for the pure ROS2<->gpu_planning translation logic.

These exercise the geometry / joint-ordering helpers in
:mod:`gpu_planning_ros2.conversions` using light stand-in objects, so they run
without a ROS graph, without ``rclpy``, and without a GPU. They guard the two
behaviours that matter for correctness: conservative primitive->OBB
over-approximation, and start/goal joint reordering.
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpu_planning_ros2 import conversions  # noqa: E402


# --- tiny attribute-bag stand-ins for the ROS message structs --------------- #
def _obj(**kw):
    return types.SimpleNamespace(**kw)


def test_box_primitive_maps_exactly():
    assert conversions.solid_primitive_to_obb(1, [0.2, 0.3, 0.4]) == (0.2, 0.3, 0.4)


def test_sphere_over_approximated_to_cube():
    # radius 0.5 -> full extent 1.0 on every axis (enclosing cube).
    assert conversions.solid_primitive_to_obb(2, [0.5]) == (1.0, 1.0, 1.0)


def test_cylinder_over_approximated():
    # [height, radius] = [2.0, 0.25] -> (0.5, 0.5, 2.0)
    assert conversions.solid_primitive_to_obb(3, [2.0, 0.25]) == (0.5, 0.5, 2.0)


def test_unknown_primitive_is_none():
    assert conversions.solid_primitive_to_obb(99, [1.0]) is None


def test_collision_objects_to_cuboids_skips_remove():
    pose = _obj(position=_obj(x=1.0, y=2.0, z=3.0),
                orientation=_obj(w=1.0, x=0.0, y=0.0, z=0.0))
    keep = _obj(id="table", operation=0,
                primitives=[_obj(type=1, dimensions=[1.0, 1.0, 0.1])],
                primitive_poses=[pose])
    drop = _obj(id="ghost", operation=1,
                primitives=[_obj(type=1, dimensions=[1.0, 1.0, 1.0])],
                primitive_poses=[pose])
    out = conversions.collision_objects_to_cuboids([keep, drop])
    assert len(out) == 1
    assert out[0]["name"] == "table"
    assert out[0]["dims"] == [1.0, 1.0, 0.1]
    assert out[0]["position"] == [1.0, 2.0, 3.0]
    assert out[0]["quaternion"] == [1.0, 0.0, 0.0, 0.0]


def test_multi_primitive_object_gets_indexed_names():
    pose = _obj(position=_obj(x=0.0, y=0.0, z=0.0),
                orientation=_obj(w=1.0, x=0.0, y=0.0, z=0.0))
    obj = _obj(id="combo", operation=0,
               primitives=[_obj(type=1, dimensions=[1, 1, 1]),
                           _obj(type=2, dimensions=[0.5])],
               primitive_poses=[pose, pose])
    out = conversions.collision_objects_to_cuboids([obj])
    assert [o["name"] for o in out] == ["combo#0", "combo#1"]


def test_extract_start_goal_reorders_to_start_names():
    start_js = _obj(name=["j1", "j2", "j3"], position=[0.1, 0.2, 0.3])
    goal_js = _obj(name=["j3", "j1", "j2"], position=[3.0, 1.0, 2.0])
    names, start, goal = conversions.extract_start_goal(start_js, goal_js)
    assert names == ["j1", "j2", "j3"]
    np.testing.assert_allclose(start, [0.1, 0.2, 0.3], rtol=0, atol=1e-6)
    np.testing.assert_allclose(goal, [1.0, 2.0, 3.0], rtol=0, atol=1e-6)


def test_extract_start_goal_requires_goal():
    start_js = _obj(name=["j1"], position=[0.0])
    goal_js = _obj(name=[], position=[])
    try:
        conversions.extract_start_goal(start_js, goal_js)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty goal")


def test_result_points_uniform_dt_and_zero_fill():
    result = _obj(trajectory=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
                  velocities=None, dt=0.05)
    pts = conversions.result_points(result)
    assert len(pts) == 3
    # third point at t = 2 * dt
    assert abs(pts[2][2] - 0.10) < 1e-9
    # zero-filled velocities, correct width
    np.testing.assert_allclose(pts[0][1], [0.0, 0.0])
