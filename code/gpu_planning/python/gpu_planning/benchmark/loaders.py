# -*- coding: utf-8 -*-
"""Loaders that normalise external benchmark formats into ``BenchProblem``.

Two sources are supported, both variants of MotionBenchMaker (MBM):

* **VAMP** (``load_vamp_problems``) — the ``.pkl`` / ``.json`` produced by
  VAMP's ``problem_tar_to_pkl_json.py``. Goals are joint configurations, so
  every valid problem is directly usable by our planners.
* **Robometrics / mpinets** (``load_robometrics_problems``) — the datasets
  shipped by the ``robometrics`` library, with geometry as ``geometrout``
  primitives. Goals here are end-effector **poses**; because this package
  ships **no IK solver**, we can only use problems that also carry a
  joint-space goal (``goal_ik``). Pose-only problems are skipped with the
  reason ``"pose_goal_no_ik"``.

Both yield the same :class:`BenchProblem`, so the runner treats them
identically. numpy only; no ``prrtc`` / torch import.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .problem import BenchObstacle, BenchProblem, euler_xyz_to_quat_wxyz


# --------------------------------------------------------------------------- #
# VAMP MBM pkl/json
# --------------------------------------------------------------------------- #
def _load_raw(path: str) -> Dict[str, Any]:
    """Load a VAMP problem file (``.json`` or pickled ``.pkl``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"benchmark file not found: {path}")
    if p.suffix == ".json":
        with p.open("r") as f:
            return json.load(f)
    with p.open("rb") as f:
        return pickle.load(f)


def _vamp_obstacles(prob: Dict[str, Any]) -> List[BenchObstacle]:
    out: List[BenchObstacle] = []
    for i, b in enumerate(prob.get("box", []) or []):
        out.append(BenchObstacle(
            kind="box",
            name=b.get("name", f"box_{i}"),
            position=np.asarray(b["position"], dtype=np.float64).reshape(3),
            quat_wxyz=euler_xyz_to_quat_wxyz(b["orientation_euler_xyz"]),
            half_extents=np.asarray(b["half_extents"], dtype=np.float64).reshape(3),
        ))
    for i, s in enumerate(prob.get("sphere", []) or []):
        out.append(BenchObstacle(
            kind="sphere",
            name=s.get("name", f"sphere_{i}"),
            position=np.asarray(s["position"], dtype=np.float64).reshape(3),
            radius=float(s["radius"]),
        ))
    for i, c in enumerate(prob.get("cylinder", []) or []):
        out.append(BenchObstacle(
            kind="cylinder",
            name=c.get("name", f"cylinder_{i}"),
            position=np.asarray(c["position"], dtype=np.float64).reshape(3),
            quat_wxyz=euler_xyz_to_quat_wxyz(c["orientation_euler_xyz"]),
            radius=float(c["radius"]),
            length=float(c["length"]),
        ))
    return out


def load_vamp_problems(
    path: str,
    scenes: Optional[Sequence[str]] = None,
    max_per_scene: Optional[int] = None,
) -> List[BenchProblem]:
    """Parse a VAMP MBM ``.pkl``/``.json`` into normalized ``BenchProblem``s.

    Structure: ``raw["problems"][scene]`` is a list of problem dicts, each with
    ``valid``, ``start`` (joint config), ``goals`` (list of joint configs), and
    ``box`` / ``sphere`` / ``cylinder`` obstacle lists. Invalid problems and
    those without a usable joint goal are skipped.
    """
    raw = _load_raw(path)
    by_scene = raw.get("problems", raw)
    if not isinstance(by_scene, dict):
        raise ValueError(
            "unexpected VAMP file layout: expected a dict of scene -> problems")

    wanted = set(scenes) if scenes else None
    out: List[BenchProblem] = []
    for scene, probs in by_scene.items():
        if wanted is not None and scene not in wanted:
            continue
        count = 0
        for idx, prob in enumerate(probs or []):
            if not prob.get("valid", True):
                continue
            goals = prob.get("goals") or []
            if not goals:
                continue
            start = np.asarray(prob["start"], dtype=np.float32).reshape(-1)
            goal = np.asarray(goals[0], dtype=np.float32).reshape(-1)
            out.append(BenchProblem(
                scene=scene, index=idx, start=start, goal=goal,
                obstacles=_vamp_obstacles(prob), source="vamp",
                meta={"problem_type": prob.get("problem", scene),
                      "n_goals": len(goals)}))
            count += 1
            if max_per_scene is not None and count >= max_per_scene:
                break
    return out


# --------------------------------------------------------------------------- #
# Robometrics / mpinets (geometrout primitives, pose goals)
# --------------------------------------------------------------------------- #

# The MBM / mpinets scenes include the fixed pedestal the arm is bolted to
# (e.g. ``cube_robot_stand``). It sits directly under ``panda_link0`` and would
# trivially collide with the base links, so every reference pipeline treats it
# via the allowed-collision matrix rather than as a plannable obstacle. This
# harness has no ACM plumbing, so we drop the mount at load time — matching how
# MBM/mpinets/VAMP effectively ignore base↔mount contact.
_ROBOT_MOUNT_TOKENS = ("robot_stand", "robot_base", "pedestal", "mount")


def _is_robot_mount(name: str) -> bool:
    low = str(name).lower()
    return any(tok in low for tok in _ROBOT_MOUNT_TOKENS)


def _attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    """First present attribute (or dict key) among ``names``."""
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _quat_wxyz_of(prim: Any) -> np.ndarray:
    """Pull a (w,x,y,z) quaternion from a geometrout primitive.

    geometrout exposes orientation as ``prim.pose.so3.wxyz`` (SE3 / SO3), or on
    older versions a direct ``quaternion`` / ``wxyz`` attribute. Falls back to
    identity.
    """
    pose = _attr(prim, ["pose"])
    if pose is not None:
        so3 = _attr(pose, ["so3", "rotation"])
        wxyz = _attr(so3 if so3 is not None else pose, ["wxyz", "q", "quaternion"])
        if wxyz is not None:
            return np.asarray(wxyz, dtype=np.float64).reshape(4)
    wxyz = _attr(prim, ["wxyz", "quaternion"])
    if wxyz is not None:
        return np.asarray(wxyz, dtype=np.float64).reshape(4)
    return np.array([1.0, 0.0, 0.0, 0.0])


def _center_of(prim: Any) -> np.ndarray:
    pose = _attr(prim, ["pose"])
    if pose is not None:
        xyz = _attr(pose, ["xyz", "translation", "pos"])
        if xyz is not None:
            return np.asarray(xyz, dtype=np.float64).reshape(3)
    c = _attr(prim, ["center", "position", "centroid"])
    return np.asarray(c, dtype=np.float64).reshape(3) if c is not None else np.zeros(3)


def _prim_to_obstacle(prim: Any, name: str) -> Optional[BenchObstacle]:
    """Dispatch a geometrout primitive to a ``BenchObstacle`` by its shape."""
    cls = type(prim).__name__.lower()
    if "sphere" in cls:
        return BenchObstacle(kind="sphere", name=name, position=_center_of(prim),
                             radius=float(_attr(prim, ["radius"], 0.0)))
    if "cylinder" in cls or "capsule" in cls:
        return BenchObstacle(
            kind="cylinder", name=name, position=_center_of(prim),
            quat_wxyz=_quat_wxyz_of(prim),
            radius=float(_attr(prim, ["radius"], 0.0)),
            length=float(_attr(prim, ["height", "length"], 0.0)))
    if "cuboid" in cls or "box" in cls:
        dims = _attr(prim, ["dims", "dimensions"])
        half = np.asarray(dims, dtype=np.float64).reshape(3) / 2.0 \
            if dims is not None else np.zeros(3)
        return BenchObstacle(kind="box", name=name, position=_center_of(prim),
                             quat_wxyz=_quat_wxyz_of(prim), half_extents=half)
    return None


def _iter_primitives(obstacles: Any) -> Iterator[Tuple[Any, str]]:
    """Yield (primitive, name) from a robometrics obstacle container.

    Handles the layouts seen across robometrics versions:

    * grouped by **singular** key — ``obstacles["cuboid"]`` / ``["cylinder"]``
      / ``["sphere"]`` (the MBM/mpinets datasets), where each group is itself a
      ``dict`` of ``name -> primitive``;
    * grouped by **plural** attribute/key — ``.cuboids`` / ``.cylinders`` /
      ``.spheres``, either a list or a name->primitive dict;
    * a flat iterable of primitives.
    """
    if obstacles is None:
        return

    def _emit(items: Any, prefix: str) -> Iterator[Tuple[Any, str]]:
        if isinstance(items, dict):
            for name, prim in items.items():
                yield prim, str(name)
        else:
            for j, prim in enumerate(items):
                yield prim, f"{prefix}_{j}"

    grouped = False
    for grp in ("cuboid", "cylinder", "sphere",
                "cuboids", "cylinders", "spheres"):
        items = _attr(obstacles, [grp])
        if items:
            grouped = True
            yield from _emit(items, grp.rstrip("s"))
    if grouped:
        return
    for j, prim in enumerate(obstacles):
        yield prim, f"obs_{j}"


def _joint_goal_of(prob: Any) -> Optional[np.ndarray]:
    """A joint-space goal from a robometrics problem, or None if pose-only."""
    gik = _attr(prob, ["goal_ik", "target_ik", "q_goal", "goal_config"])
    if gik is None:
        return None
    arr = np.asarray(gik, dtype=np.float32)
    if arr.ndim >= 2:            # list of IK solutions -> first
        arr = arr[0]
    return arr.reshape(-1) if arr.size else None


def load_robometrics_problems(
    dataset: str = "mbm",
    scenes: Optional[Sequence[str]] = None,
    max_per_scene: Optional[int] = None,
) -> Tuple[List[BenchProblem], Dict[str, int]]:
    """Load a robometrics dataset, keeping only joint-goal problems.

    ``dataset`` is one of ``"mbm"`` (motion_benchmaker), ``"mpinets"``,
    ``"demo"``. Returns ``(problems, skipped)`` where ``skipped`` counts the
    reasons problems were dropped (notably ``"pose_goal_no_ik"``). Requires the
    ``robometrics`` package to be importable.
    """
    try:
        from robometrics.datasets import demo, motion_benchmaker, mpinets
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "robometrics.datasets is not importable; install "
            "`python3 -m pip install git+https://github.com/fishbotics/robometrics.git geometrout` "
            "to use --source robometrics. "
            "The unrelated PyPI package named robometrics does not supply "
            "these datasets.") from e

    loaders = {"mbm": motion_benchmaker, "motion_benchmaker": motion_benchmaker,
               "mpinets": mpinets, "demo": demo}
    if dataset not in loaders:
        raise ValueError(f"unknown robometrics dataset {dataset!r}; "
                         f"choose from {sorted(loaders)}")
    raw = loaders[dataset]()

    wanted = set(scenes) if scenes else None
    skipped: Dict[str, int] = {"pose_goal_no_ik": 0, "no_start": 0}
    out: List[BenchProblem] = []
    for scene, probs in raw.items():
        if wanted is not None and scene not in wanted:
            continue
        count = 0
        for idx, prob in enumerate(probs or []):
            start = _attr(prob, ["q0", "start", "start_config"])
            if start is None:
                skipped["no_start"] += 1
                continue
            goal = _joint_goal_of(prob)
            if goal is None:
                skipped["pose_goal_no_ik"] += 1
                continue
            obstacles = [
                o for o in (
                    _prim_to_obstacle(prim, name)
                    for prim, name in _iter_primitives(_attr(prob, ["obstacles"]))
                    if not _is_robot_mount(name)
                ) if o is not None
            ]
            out.append(BenchProblem(
                scene=scene, index=idx,
                start=np.asarray(start, dtype=np.float32).reshape(-1),
                goal=goal, obstacles=obstacles, source="robometrics",
                meta={"dataset": dataset}))
            count += 1
            if max_per_scene is not None and count >= max_per_scene:
                break
    return out, skipped
