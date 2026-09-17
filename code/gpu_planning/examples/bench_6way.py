# -*- coding: utf-8 -*-
"""bench_6way — run all six substrate planners on one shared scene.

This is the reference example for the package: it builds a single obstacle
world (:class:`~gpu_planning.PlanningScene`) and a single joint-space
start→goal request, then plans it with each of the six substrate-consistent
planners in turn — pRRTC, MIT*, MHA*, wPA*SE, STOMP, CHOMP — and prints a
side-by-side comparison. Every planner is its own :class:`PlannerBase` subclass
loading its own ``config/planners/<name>.yaml``; there is no
``extra_params["solver"]`` dispatch. They all share the same GPU collision
substrate, robot model, and time parameterisation, so the trajectories are
directly comparable.

Requires the compiled ``prrtc`` pybind module (and a CUDA device) on the
``PYTHONPATH``. Run from anywhere:

    python examples/bench_6way.py
    python examples/bench_6way.py --asset config/robots/panda.yaml --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running directly (`python examples/bench_6way.py`) without installing:
# add the package's `python/` source dir to the import path.
_PKG_PY = Path(__file__).resolve().parents[1] / "python"
if _PKG_PY.is_dir() and str(_PKG_PY) not in sys.path:
    sys.path.insert(0, str(_PKG_PY))

from gpu_planning import (
    CHOMPPlanner,
    MHAStarPlanner,
    MITStarPlanner,
    PRRTCPlanner,
    STOMPPlanner,
    WPASEPlanner,
    PlanningRequest,
    PlanningScene,
    load_robot_asset,
)

# The six planners, in paradigm order (sampling, sampling-optimal, search,
# parallel-search, optimisation-free, optimisation-gradient). Each loads its
# own default YAML.
PLANNERS = [
    ("pRRTC", PRRTCPlanner),
    ("MIT*", MITStarPlanner),
    ("MHA*", MHAStarPlanner),
    ("wPA*SE", WPASEPlanner),
    ("STOMP", STOMPPlanner),
    ("CHOMP", CHOMPPlanner),
]


def build_scene(asset_path, device):
    """A small panda cell: a table slab below and a pillar in front.

    Returns ``(scene, n_obstacles)``. ``scene.n_obbs`` reports 0 until the GPU
    collision data is built (lazily, inside the first planner's ``initialize``),
    so we return the count of obstacles we staged for an accurate banner.
    """
    scene = PlanningScene.from_robot_asset(asset_path, device=device)
    # Table top: a thin slab lowered to z=-0.1 so it clears the robot base
    # (which sits at z=0 — a slab at z=0 would collide with it at the start).
    scene.add_cuboid("table", dims=(1.2, 1.2, 0.05), position=(0.5, 0.0, -0.1))
    # A pillar the arm must route around, tucked into a side pocket that
    # neither the 'ready' start nor the goal EE pose intersects.
    scene.add_cuboid("box", dims=(0.1, 0.1, 0.4), position=(0.45, -0.2, 0.5))
    return scene, 2


def build_request(asset):
    """Franka 'ready' pose → a reach to the side, both well inside limits."""
    lower = np.asarray(asset["joint_lower"], dtype=np.float32)
    upper = np.asarray(asset["joint_upper"], dtype=np.float32)
    start = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
                     dtype=np.float32)
    goal = np.array([0.6, 0.35, -0.4, -1.9, 0.25, 1.95, 0.9], dtype=np.float32)
    # Clamp defensively in case the asset limits are tighter than assumed.
    start = np.clip(start, lower, upper)
    goal = np.clip(goal, lower, upper)
    return PlanningRequest(
        robot_id=asset["name"],
        start_joint_state=start,
        target_joint_state=goal,
        velocity_scaling=1.0,
        acceleration_scaling=1.0,
    )


def _cost(result):
    """Extract a comparable path cost from the planner-specific extra_info."""
    info = result.extra_info or {}
    for key in ("cost", "final_total_cost"):
        if key in info:
            return float(info[key])
    return float("nan")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default=None,
                        help="robot YAML asset (default: bundled panda)")
    parser.add_argument("--device", default="cuda",
                        help="torch device for scene tensors (default: cuda)")
    args = parser.parse_args(argv)

    try:
        import prrtc  # noqa: F401
    except ImportError:
        print("ERROR: the compiled `prrtc` module is not importable — build "
              "the pybind extension in cpp/ first (needs a CUDA toolchain).",
              file=sys.stderr)
        return 1

    asset = load_robot_asset(args.asset)
    scene, n_obstacles = build_scene(args.asset, args.device)
    request = build_request(asset)

    print(f"robot={asset['name']} n_dof={asset['n_dof']} "
          f"obstacles={n_obstacles} device={args.device}")
    print(f"start={np.round(request.start_joint_state, 3).tolist()}")
    print(f"goal ={np.round(request.target_joint_state, 3).tolist()}\n")

    header = f"{'planner':8s} {'ok':3s} {'status':18s} " \
             f"{'solve_ms':>9s} {'gpu_ms':>8s} {'wpts':>5s} {'t_traj':>7s} {'cost':>9s}"
    print(header)
    print("-" * len(header))

    for label, cls in PLANNERS:
        planner = cls()  # discovers config/planners/<name>.yaml
        if not planner.initialize(scene, asset_path=args.asset):
            print(f"{label:8s} {'-':3s} {'init-failed':18s}")
            continue
        try:
            result = planner.plan(request)
        finally:
            planner.shutdown()

        info = result.extra_info or {}
        gpu_ms = info.get("wall_time_ms", float("nan"))
        wpts = info.get("geometric_waypoints", result.num_waypoints)
        print(f"{label:8s} "
              f"{('yes' if result.success else 'no'):3s} "
              f"{result.status:18s} "
              f"{result.solve_time * 1e3:9.1f} "
              f"{gpu_ms:8.1f} "
              f"{wpts:5d} "
              f"{result.total_time:7.2f} "
              f"{_cost(result):9.4g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
