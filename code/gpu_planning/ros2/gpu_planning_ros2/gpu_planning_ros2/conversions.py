# -*- coding: utf-8 -*-
"""Translation between ROS 2 messages and ``gpu_planning`` dataclasses.

Kept separate from the node so the geometry / ordering logic is unit-testable
without a running ROS graph. The functions that touch message *types* take the
already-deserialised message objects as plain arguments; only field access is
used, so no message class needs to be imported here. Everything is built on
ROS 2 common_interfaces types (``sensor_msgs``, ``shape_msgs``, ``geometry_msgs``,
``trajectory_msgs``) plus this package's own ``gpu_planning_msgs`` — there is no
MoveIt dependency.

Design choices that mirror the paper's collision substrate:

* The substrate's evaluated path is sphere-vs-OBB, so every collision primitive
  is reduced to an axis/orientation-preserving **oriented bounding box**. A BOX
  maps exactly; a SPHERE / CYLINDER is conservatively over-approximated to its
  enclosing box (identical to the 800-problem MotionBenchMaker harness).
* Joint order is taken from the request's ``start`` JointState ``name`` list and
  the goal vector is reordered to match it, so the vector handed to the planner
  is consistent with the robot model's joint order.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# shape_msgs/SolidPrimitive.type enum values (stable ROS constants).
_BOX = 1
_SPHERE = 2
_CYLINDER = 3
_CONE = 4


def solid_primitive_to_obb(
    prim_type: int, dimensions: Sequence[float],
) -> Optional[Tuple[float, float, float]]:
    """Return full-extent ``(dx, dy, dz)`` of the enclosing box for a primitive.

    ``None`` for an unsupported / empty primitive (caller should skip it).
    Over-approximation is intentional and conservative: the enclosing box never
    under-reports occupancy, so a plan that is collision-free against the box is
    collision-free against the true primitive.
    """
    d = list(dimensions)
    if prim_type == _BOX and len(d) >= 3:
        return (float(d[0]), float(d[1]), float(d[2]))
    if prim_type == _SPHERE and len(d) >= 1:
        s = 2.0 * float(d[0])  # radius -> full extent on every axis
        return (s, s, s)
    if prim_type == _CYLINDER and len(d) >= 2:
        # dimensions = [height, radius]
        h = float(d[0])
        diam = 2.0 * float(d[1])
        return (diam, diam, h)
    if prim_type == _CONE and len(d) >= 2:
        h = float(d[0])
        diam = 2.0 * float(d[1])
        return (diam, diam, h)
    return None


def _pose_to_pos_quat(pose) -> Tuple[List[float], List[float]]:
    """geometry_msgs/Pose -> (position[x,y,z], quaternion[w,x,y,z])."""
    p = pose.position
    o = pose.orientation
    return ([float(p.x), float(p.y), float(p.z)],
            [float(o.w), float(o.x), float(o.y), float(o.z)])


def collision_objects_to_cuboids(collision_objects) -> List[Dict]:
    """Flatten gpu_planning_msgs/CollisionObject[] into ``scene.update_world`` dicts.

    Each output dict is ``{"name", "dims":[dx,dy,dz], "position":[x,y,z],
    "quaternion":[w,x,y,z]}``. One entry per primitive; a multi-primitive
    object yields ``<id>#<k>`` names. Objects with ``operation == REMOVE`` (1)
    are skipped (the node rebuilds the whole world each scene message, so a
    removed object simply does not appear).
    """
    out: List[Dict] = []
    for obj in collision_objects:
        operation = int(getattr(obj, "operation", 0))
        if operation == 1:  # REMOVE
            continue
        prims = list(getattr(obj, "primitives", []))
        poses = list(getattr(obj, "primitive_poses", []))
        obj_id = getattr(obj, "id", "obj")
        for k, prim in enumerate(prims):
            dims = solid_primitive_to_obb(int(prim.type), prim.dimensions)
            if dims is None or k >= len(poses):
                continue
            position, quat = _pose_to_pos_quat(poses[k])
            name = obj_id if len(prims) == 1 else f"{obj_id}#{k}"
            out.append({
                "name": name,
                "dims": list(dims),
                "position": position,
                "quaternion": quat,
            })
    return out


def _joint_state_to_map(joint_state) -> Dict[str, float]:
    names = list(getattr(joint_state, "name", []))
    positions = list(getattr(joint_state, "position", []))
    return {n: float(p) for n, p in zip(names, positions)}


def extract_start_goal(
    start_js, goal_js,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Pull ``(joint_names, start_vec, goal_vec)`` from two JointState messages.

    Joint order is defined by ``start_js.name``. The goal is read from
    ``goal_js`` (name+position) and reordered to that same joint-name list. A
    joint named only in the goal is appended to the order rather than dropped;
    a joint absent from the goal falls back to its start value (i.e. it is held).

    Raises ``ValueError`` if the start has no named joints or the goal is empty.
    """
    joint_names = list(getattr(start_js, "name", []))
    start_map = _joint_state_to_map(start_js)
    if not joint_names:
        raise ValueError("start JointState has no named joints")

    goal_map = _joint_state_to_map(goal_js)
    if not goal_map:
        raise ValueError("goal JointState has no named joints")

    for n in goal_map:
        if n not in joint_names:
            joint_names.append(n)

    start_vec = np.array([start_map.get(n, 0.0) for n in joint_names],
                         dtype=np.float32)
    goal_vec = np.array([goal_map.get(n, start_map.get(n, 0.0))
                         for n in joint_names], dtype=np.float32)
    return joint_names, start_vec, goal_vec


def result_points(result) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Yield ``(positions, velocities, time_from_start_sec)`` per waypoint.

    Uniform ``result.dt`` spacing; velocities are zero-filled if the planner did
    not return a velocity profile. Pure numpy — no ROS types — so the node just
    wraps these into JointTrajectoryPoint messages.
    """
    traj = np.asarray(result.trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] == 0:
        return []
    n, dof = traj.shape
    vel = result.velocities
    if vel is not None:
        vel = np.asarray(vel, dtype=np.float64)
        if vel.shape != traj.shape:
            vel = None
    dt = float(getattr(result, "dt", 0.02)) or 0.02
    pts: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(n):
        v = vel[i] if vel is not None else np.zeros(dof)
        pts.append((traj[i], v, i * dt))
    return pts
