"""Validates that post-processed (shortcut) paths are actually collision-free.

shortcut_cc_kernel decides which skip-edges are safe. If its FK disagrees with
the substrate's tree walk, it will happily splice in an edge that passes
through an obstacle -- and nothing else in the pipeline re-checks the result.
Panda is a serial chain and cannot detect that; Baxter's second arm hangs off
the root and does.

Interpolates every returned path segment finely and re-checks each waypoint
with the independent NumPy tree FK + sphere/OBB test from
tests/test_tree_fk_vs_cpu.py.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

from test_tree_fk_vs_cpu import fk_spheres, in_collision  # noqa: E402

from gpu_planning import PRRTCPlanner  # noqa: E402
from gpu_planning.base.planner_base import PlanningRequest  # noqa: E402
from gpu_planning.benchmark import load_vamp_problems, obstacles_to_obbs  # noqa: E402
from gpu_planning.substrate import PlanningScene, load_robot_asset  # noqa: E402

STEP = 0.02          # rad between re-checked waypoints


def check(asset_path, problems_path, planner, cfg_note):
    asset = load_robot_asset(asset_path)
    scene = PlanningScene.from_robot_asset(asset_path, device="cuda")
    assert planner.initialize(scene, asset_path=asset_path)

    probs = load_vamp_problems(problems_path, max_per_scene=5)
    n_solved = n_bad_wp = n_bad_paths = 0
    for p in probs:
        obb_dicts = obstacles_to_obbs(p.obstacles)
        scene.update_world(obb_dicts)
        obbs = [(np.asarray(d["dims"], np.float64),
                 np.asarray(d["position"], np.float64),
                 np.asarray(d.get("quaternion", (1, 0, 0, 0)), np.float64))
                for d in obb_dicts]

        res = planner.plan(PlanningRequest(
            robot_id=asset["name"],
            start_joint_state=np.asarray(p.start, np.float32),
            target_joint_state=np.asarray(p.goal, np.float32),
            velocity_scaling=1.0, acceleration_scaling=1.0))
        if not res.success:
            continue
        n_solved += 1

        path = np.asarray(res.trajectory, np.float64)
        bad_here = 0
        for a, b in zip(path[:-1], path[1:]):
            n_steps = max(1, int(np.ceil(np.linalg.norm(b - a) / STEP)))
            for k in range(n_steps + 1):
                q = a + (b - a) * (k / n_steps)
                if in_collision(fk_spheres(asset, q), obbs):
                    bad_here += 1
        if bad_here:
            n_bad_paths += 1
            n_bad_wp += bad_here
            print(f"  IN-COLLISION {p.scene}#{p.index}: {bad_here} waypoints, "
                  f"{len(path)} path points")

    planner.shutdown()
    print(f"{cfg_note}: solved={n_solved} bad_paths={n_bad_paths} "
          f"bad_waypoints={n_bad_wp}")
    return n_bad_paths


def main():
    problems_path = os.environ.get("PRISM_BAXTER_PROBLEMS")
    if not problems_path or not Path(problems_path).is_file():
        print("SKIP: set PRISM_BAXTER_PROBLEMS to a Baxter MBM JSON file")
        return 0
    bad = check(str(ROOT / "config" / "robots" / "baxter.yaml"),
                problems_path,
                PRRTCPlanner(config={"shortcut_path": True}),
                "baxter/pRRTC shortcut=on")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
