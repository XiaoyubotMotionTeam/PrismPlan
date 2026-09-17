# -*- coding: utf-8 -*-
"""Normalized benchmark problem types + conservative OBB approximation.

The benchmark harness ingests motion-planning problems from several external
formats (MotionBenchMaker via VAMP tarballs, Robometrics/mpinets pickles).
Those formats describe obstacles with three primitive shapes — boxes, spheres,
and cylinders — but this package's :class:`~gpu_planning.PlanningScene`
collision substrate ingests **oriented bounding boxes (OBBs) only**.

We therefore normalise every external problem into the small dataclasses here
and then over-approximate every non-box primitive with a single enclosing OBB
(:func:`obstacles_to_obbs`). Over-approximation is conservative: the OBB fully
contains the primitive, so the substrate can never *miss* a real collision (it
may only report a collision slightly early). All six planners see the exact
same OBB world, so their results stay internally comparable — which is the
property the substrate-consistency thesis actually needs. (cuRobo makes the
same box-approximation choice on this dataset, so there is precedent.)

Pure Python + numpy; imports no ``prrtc`` and no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Orientation helper
# --------------------------------------------------------------------------- #
def euler_xyz_to_quat_wxyz(euler: Sequence[float]) -> np.ndarray:
    """Convert an intrinsic/extrinsic XYZ Euler triple to a (w,x,y,z) quat.

    VAMP's MBM export stores obstacle orientation as ``orientation_euler_xyz``.
    We interpret it as the fixed-axis (extrinsic) XYZ convention, i.e. the
    rotation matrix ``R = Rz @ Ry @ Rx`` applied to a column vector. This is
    the SciPy ``Rotation.from_euler("xyz", ...)`` convention, which is what the
    VAMP generator uses. The equivalent quaternion is the product
    ``q = qz * qy * qx``.

    (Documented explicitly so it can be re-validated against a live VAMP
    problem on the GPU machine — a wrong convention would rotate obstacles but
    would still yield a valid, conservative world, just a different one.)
    """
    e = np.asarray(euler, dtype=np.float64).reshape(3)
    hx, hy, hz = e * 0.5
    cx, sx = np.cos(hx), np.sin(hx)
    cy, sy = np.cos(hy), np.sin(hy)
    cz, sz = np.cos(hz), np.sin(hz)

    # q = qz (*) qy (*) qx, each a (w, x, y, z) unit quaternion about one axis.
    qx = np.array([cx, sx, 0.0, 0.0])
    qy = np.array([cy, 0.0, sy, 0.0])
    qz = np.array([cz, 0.0, 0.0, sz])
    q = _quat_mul(_quat_mul(qz, qy), qx)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w,x,y,z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


# --------------------------------------------------------------------------- #
# Normalized obstacle + problem
# --------------------------------------------------------------------------- #
@dataclass
class BenchObstacle:
    """One primitive obstacle in a normalized, source-agnostic form.

    ``kind`` is ``"box"``, ``"sphere"`` or ``"cylinder"``. Fields not relevant
    to a kind are left at their defaults:

    * box      — ``position``, ``quat_wxyz``, ``half_extents`` (3,)
    * sphere   — ``position``, ``radius``
    * cylinder — ``position``, ``quat_wxyz``, ``radius``, ``length`` (axis = the
      obstacle-local +z direction, matching VAMP / geometrout)
    """

    kind: str
    name: str = ""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    half_extents: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.0
    length: float = 0.0


@dataclass
class BenchProblem:
    """A normalized joint-space planning query with its obstacle world.

    ``start`` and ``goal`` are joint configurations (radians). External formats
    that only provide an end-effector pose goal (with no joint IK) cannot be
    represented here — the loader skips them, because this package ships no IK
    solver.
    """

    scene: str
    index: int
    start: np.ndarray
    goal: np.ndarray
    obstacles: List[BenchObstacle] = field(default_factory=list)
    source: str = ""
    meta: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Conservative OBB over-approximation
# --------------------------------------------------------------------------- #
def obstacle_to_obb(obs: BenchObstacle) -> Dict[str, object]:
    """Over-approximate one primitive with a single enclosing OBB dict.

    Returns a dict in the shape :meth:`PlanningScene.update_world` accepts:
    ``{"name", "dims":[dx,dy,dz], "position":[x,y,z], "quaternion":[w,x,y,z]}``
    where ``dims`` are **full** extents.

    * box      — exact: ``dims = 2 * half_extents`` in the box's own frame.
    * sphere   — axis-aligned cube ``dims = [2r, 2r, 2r]``, identity orientation
      (a sphere's enclosing box is orientation-free).
    * cylinder — oriented box ``dims = [2r, 2r, length]`` aligned to the
      cylinder frame, since the cylinder axis is local +z. This is the tight
      enclosing box of the cylinder.
    """
    pos = np.asarray(obs.position, dtype=np.float64).reshape(3)
    if obs.kind == "box":
        dims = 2.0 * np.asarray(obs.half_extents, dtype=np.float64).reshape(3)
        quat = np.asarray(obs.quat_wxyz, dtype=np.float64).reshape(4)
    elif obs.kind == "sphere":
        r = float(obs.radius)
        dims = np.array([2.0 * r, 2.0 * r, 2.0 * r])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    elif obs.kind == "cylinder":
        r = float(obs.radius)
        dims = np.array([2.0 * r, 2.0 * r, float(obs.length)])
        quat = np.asarray(obs.quat_wxyz, dtype=np.float64).reshape(4)
    else:
        raise ValueError(f"unknown obstacle kind: {obs.kind!r}")

    return {
        "name": obs.name or obs.kind,
        "dims": dims.tolist(),
        "position": pos.tolist(),
        "quaternion": quat.tolist(),
    }


def obstacles_to_obbs(obstacles: Sequence[BenchObstacle]) -> List[Dict[str, object]]:
    """Convert a list of primitives to OBB dicts for ``update_world``.

    Names are de-duplicated (``update_world`` keys obstacles by name, so
    collisions would silently drop obstacles) by suffixing an index.
    """
    out: List[Dict[str, object]] = []
    seen: Dict[str, int] = {}
    for i, obs in enumerate(obstacles):
        d = obstacle_to_obb(obs)
        base = str(d["name"])
        if base in seen:
            seen[base] += 1
            d["name"] = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            d["name"] = f"{base}_0" if base else f"obs_{i}"
        out.append(d)
    return out
