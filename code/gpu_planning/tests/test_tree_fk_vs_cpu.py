"""Validates the GPU tree FK + scene check against an independent NumPy
reference, on the branching 14-DoF Baxter tree.

The GPU path (fk_runtime -> scene_collision_check_runtime, reached through
prrtc.check_collision_scene) must agree with a straightforward CPU tree walk
plus sphere-vs-OBB test. Panda is a serial chain, so it cannot detect a broken
tree walk; Baxter's second arm hangs off the root and does.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpu_planning import PRRTCPlanner  # noqa: E402
from gpu_planning.benchmark import load_vamp_problems, obstacles_to_obbs  # noqa: E402
from gpu_planning.substrate import PlanningScene, load_robot_asset  # noqa: E402

X_PRISM, Y_PRISM, Z_PRISM, X_ROT, Y_ROT, Z_ROT = range(6)


def joint_motion(jtype, q):
    T = np.eye(4)
    c, s = np.cos(q), np.sin(q)
    if jtype == X_ROT:
        T[1:3, 1:3] = [[c, -s], [s, c]]
    elif jtype == Y_ROT:
        T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    elif jtype == Z_ROT:
        T[0:2, 0:2] = [[c, -s], [s, c]]
    elif jtype in (X_PRISM, Y_PRISM, Z_PRISM):
        T[jtype, 3] = q
    return T


def fk_spheres(asset, q):
    """World-frame (x,y,z,r) for every collision sphere."""
    n_joints = asset["n_joints"]
    ft = np.asarray(asset["fixed_transforms"], np.float64).reshape(n_joints, 4, 4)
    frames = [None] * n_joints
    for i in asset["dfs_order"]:
        if asset["joint_parents"][i] == i:          # root
            frames[i] = np.eye(4)
            continue
        dof = asset["joint_id_to_dof"][i]
        step = ft[i] @ joint_motion(asset["joint_types"][i],
                                    q[dof] if dof >= 0 else 0.0)
        frames[i] = frames[asset["joint_parents"][i]] @ step

    sph = np.asarray(asset["spheres_flat"], np.float64).reshape(-1, 4)
    out = np.empty_like(sph)
    for s, (j, (x, y, z, r)) in enumerate(zip(asset["sphere_to_joint"], sph)):
        out[s, :3] = (frames[j] @ np.array([x, y, z, 1.0]))[:3]
        out[s, 3] = r
    return out


def quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def in_collision(spheres, obbs):
    for dims, pos, quat in obbs:
        R = quat_to_R(*quat)
        local = (spheres[:, :3] - pos) @ R          # world -> box frame
        d = np.abs(local) - dims / 2.0
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
        inside = np.minimum(d.max(axis=1), 0.0)
        if np.any(outside + inside <= spheres[:, 3]):
            return True
    return False


def main():
    problems_path = os.environ.get("PRISM_BAXTER_PROBLEMS")
    if not problems_path or not Path(problems_path).is_file():
        print("SKIP: set PRISM_BAXTER_PROBLEMS to a Baxter MBM JSON file")
        return 0
    ASSET = str(ROOT / "config" / "robots" / "baxter.yaml")
    asset = load_robot_asset(ASSET)
    lo = np.asarray(asset["joint_lower"], np.float64)
    hi = np.asarray(asset["joint_upper"], np.float64)

    probs = load_vamp_problems(problems_path, max_per_scene=5)
    scene = PlanningScene.from_robot_asset(ASSET, device="cuda")
    planner = PRRTCPlanner()
    assert planner.initialize(scene, asset_path=ASSET)

    rng = np.random.default_rng(0)
    n_checked = n_bad = n_coll = 0
    for p in probs:
        obb_dicts = obstacles_to_obbs(p.obstacles)
        scene.update_world(obb_dicts)
        obbs = [(np.asarray(d["dims"], np.float64),
                 np.asarray(d["position"], np.float64),
                 np.asarray(d.get("quaternion", (1, 0, 0, 0)), np.float64))
                for d in obb_dicts]

        configs = np.vstack([
            np.asarray(p.start, np.float32),
            np.asarray(p.goal, np.float32),
            (lo + rng.random((30, len(lo))) * (hi - lo)).astype(np.float32),
        ])
        with planner._solve_context() as sc:
            gpu = planner._preflight(sc, configs)

        for cfg, g in zip(configs, gpu):
            cpu = in_collision(fk_spheres(asset, cfg.astype(np.float64)), obbs)
            n_checked += 1
            n_coll += int(cpu)
            if bool(g) != cpu:
                n_bad += 1
                if n_bad <= 5:
                    print(f"  MISMATCH {p.scene}#{p.index}: gpu={bool(g)} cpu={cpu}")

    planner.shutdown()
    print(f"\nchecked={n_checked} in-collision(cpu)={n_coll} mismatches={n_bad}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    sys.exit(main())
