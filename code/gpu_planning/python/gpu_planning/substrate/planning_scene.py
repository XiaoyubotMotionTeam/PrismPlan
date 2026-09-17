# -*- coding: utf-8 -*-
"""Self-contained collision world for the GPU substrate.

This is the slim replacement for the production ``GPUPlanningScene``. It owns
the GPU tensors describing the obstacle world in exactly the layout the CUDA
collision kernels expect (see ``cpp/src/collision/scene_collision_data.hh``) and
hands out a ``prrtc.SceneCollisionDataBuilder`` on demand.

What it deliberately does NOT do (all removed from the production version):
the full motion-generation / IK / kinematics stack, the multi-resolution
voxel manager, the pointcloud→voxel bridge, robot self-filtering, and the
externally-derived allowed-collision-matrix. The robot kinematic/collision model
is supplied separately by :mod:`gpu_planning.substrate.robot_model_loader`; this
class is only the obstacle world.

Obstacles are oriented bounding boxes (OBBs). The kernels also support ESDF
voxel layers, but the self-contained package ships only the OBB path (voxel
pointers are left null → the kernel simply checks no voxel layers).

Collision-tensor conventions (must match ``SceneCollisionData``):
  - ``obb_dims[i]``  = ``[dx, dy, dz, 0]``  full extents (not half).
  - ``obb_pose[i]``  = ``[tx, ty, tz, qw, qx, qy, qz, 0]``  the INVERSE pose
    ``b_T_w`` (world→OBB-local): ``p_local = R(q_stored) * p_world + t``.
  - ``obb_enable[i]`` = 0 or 1 (uint8).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .robot_model_loader import load_robot_asset

_QuatWXYZ = Tuple[float, float, float, float]


def _quat_rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by unit quaternion ``q`` (w, x, y, z).

    Matches the kernel's ``transform_sphere_to_local`` rotation:
    ``v' = v + 2w(qxyz x v) + 2 qxyz x (qxyz x v)``.
    """
    w = q_wxyz[0]
    u = q_wxyz[1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _inverse_pose(position: np.ndarray, quat_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(t, q_stored)`` of the inverse (world→local) transform.

    Given a box placed at world position ``P`` with orientation ``Q`` (local→world),
    the stored inverse pose is ``q_stored = conj(Q)`` and
    ``t = -R(q_stored) * P`` so that ``p_local = R(q_stored) * p_world + t``.
    """
    q_conj = np.array(
        [quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]], dtype=np.float64
    )
    t = -_quat_rotate(q_conj, position.astype(np.float64))
    return t, q_conj


class PlanningScene:
    """Self-contained OBB collision world for the 5 substrate planners."""

    def __init__(self, num_joints: int, robot_id: str = "robot",
                 device: str = "cuda"):
        self._num_joints = int(num_joints)
        self._robot_id = str(robot_id)
        self._device = device

        # name -> (dims(3,), position(3,), quat_wxyz(4,))  all float64 host arrays
        self._cuboids: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Guards obstacle mutation vs. tensor build/kernel use.
        self._voxel_plan_lock = threading.RLock()

        self._dirty = True
        self._torch = None
        # Device tensors + GC refs (built lazily under the lock).
        self._obb_dims_t = None
        self._obb_pose_t = None
        self._obb_enable_t = None
        self._n_obbs = 0

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_robot_asset(cls, asset_path: Optional[str] = None,
                         device: str = "cuda") -> "PlanningScene":
        """Build a scene sized to a robot YAML asset (name + n_dof)."""
        asset = load_robot_asset(asset_path)
        return cls(num_joints=asset["n_dof"], robot_id=asset["name"],
                   device=device)

    # ------------------------------------------------------------------ #
    # Read-only properties consumed by the planners
    # ------------------------------------------------------------------ #
    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def n_obbs(self) -> int:
        return self._n_obbs

    @property
    def current_obstacles(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for name, (dims, pos, quat) in self._cuboids.items():
            out.append({
                "name": name,
                "type": "cuboid",
                "dims": dims.tolist(),
                "position": pos.tolist(),
                "quaternion": quat.tolist(),  # w, x, y, z
            })
        return out

    # ------------------------------------------------------------------ #
    # Obstacle API
    # ------------------------------------------------------------------ #
    def add_cuboid(self, name: str, dims: Sequence[float],
                   position: Sequence[float],
                   quaternion: _QuatWXYZ = (1.0, 0.0, 0.0, 0.0)) -> None:
        """Add / replace an oriented box. ``quaternion`` is (w, x, y, z)."""
        dims_a = np.asarray(dims, dtype=np.float64).reshape(3)
        pos_a = np.asarray(position, dtype=np.float64).reshape(3)
        quat_a = np.asarray(quaternion, dtype=np.float64).reshape(4)
        n = np.linalg.norm(quat_a)
        if n > 0:
            quat_a = quat_a / n
        with self._voxel_plan_lock:
            self._cuboids[name] = (dims_a, pos_a, quat_a)
            self._dirty = True

    def remove_cuboid(self, name: str) -> bool:
        with self._voxel_plan_lock:
            existed = self._cuboids.pop(name, None) is not None
            if existed:
                self._dirty = True
            return existed

    def clear(self) -> None:
        with self._voxel_plan_lock:
            if self._cuboids:
                self._cuboids.clear()
                self._dirty = True

    def update_world(self, obstacles: List[Dict[str, Any]]) -> None:
        """Replace all obstacles from a list of cuboid dicts.

        Each dict: ``{"name", "dims":[dx,dy,dz], "position":[x,y,z],
        "quaternion":[w,x,y,z] (optional)}``. A 7-element ``"pose"``
        ``[x,y,z,qw,qx,qy,qz]`` is also accepted in place of position+quaternion.
        """
        with self._voxel_plan_lock:
            self._cuboids.clear()
            for i, obs in enumerate(obstacles):
                name = obs.get("name", f"obs_{i}")
                dims = obs["dims"]
                if "pose" in obs:
                    pose = obs["pose"]
                    position = pose[0:3]
                    quaternion = pose[3:7]
                else:
                    position = obs["position"]
                    quaternion = obs.get("quaternion", (1.0, 0.0, 0.0, 0.0))
                dims_a = np.asarray(dims, dtype=np.float64).reshape(3)
                pos_a = np.asarray(position, dtype=np.float64).reshape(3)
                quat_a = np.asarray(quaternion, dtype=np.float64).reshape(4)
                nrm = np.linalg.norm(quat_a)
                if nrm > 0:
                    quat_a = quat_a / nrm
                self._cuboids[name] = (dims_a, pos_a, quat_a)
            self._dirty = True

    # ------------------------------------------------------------------ #
    # Collision-data supply
    # ------------------------------------------------------------------ #
    def _rebuild_tensors(self) -> None:
        """(Re)build device OBB tensors from the current cuboid set."""
        if self._torch is None:
            import torch
            self._torch = torch
        torch = self._torch

        n = len(self._cuboids)
        self._n_obbs = n
        if n == 0:
            self._obb_dims_t = None
            self._obb_pose_t = None
            self._obb_enable_t = None
            self._dirty = False
            return

        dims = np.zeros((n, 4), dtype=np.float32)
        pose = np.zeros((n, 8), dtype=np.float32)
        enable = np.ones((n,), dtype=np.uint8)
        for i, (d, p, q) in enumerate(self._cuboids.values()):
            dims[i, 0:3] = d
            t, q_stored = _inverse_pose(p, q)
            pose[i, 0:3] = t
            pose[i, 3:7] = q_stored  # qw, qx, qy, qz
        dev = torch.device(self._device)
        self._obb_dims_t = torch.from_numpy(dims).to(dev).contiguous()
        self._obb_pose_t = torch.from_numpy(pose).to(dev).contiguous()
        self._obb_enable_t = torch.from_numpy(enable).to(dev).contiguous()
        self._dirty = False

    def build_scene_collision_data(self):
        """Return a filled ``prrtc.SceneCollisionDataBuilder`` (or None if empty).

        Must be called with ``self._voxel_plan_lock`` held for the whole span in
        which the returned builder / device tensors are used by the kernel; the
        planner base does this. Device tensors are retained on ``self`` so they
        outlive the kernel call.
        """
        import prrtc

        if self._dirty:
            self._rebuild_tensors()
        if self._n_obbs == 0:
            return None

        builder = prrtc.SceneCollisionDataBuilder()
        builder.obb_dims_ptr = self._obb_dims_t.data_ptr()
        builder.obb_pose_ptr = self._obb_pose_t.data_ptr()
        builder.obb_enable_ptr = self._obb_enable_t.data_ptr()
        builder.n_obbs = self._n_obbs
        # Voxel / ESDF and ACM pointers intentionally left null:
        #   n_voxel_layers = 0  → no ESDF layers checked
        #   sphere_acm_mask* = null → all spheres check all OBBs.
        return builder
