"""Export a robot asset + MBM problem set into a flat text file the standalone
OMPL MIT* reference bench (tools/ompl_mitstar_bench.cpp) reads.

Keeping the C++ side free of YAML/JSON parsing means the reference planner
consumes exactly the arrays our own loader produces, so any success-rate gap is
attributable to the planner and not to a divergent model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_PKG_PY = Path(__file__).resolve().parents[1] / "python"
if _PKG_PY.is_dir() and str(_PKG_PY) not in sys.path:
    sys.path.insert(0, str(_PKG_PY))

from gpu_planning.benchmark import load_vamp_problems, obstacles_to_obbs  # noqa: E402
from gpu_planning.substrate import load_robot_asset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--max-per-scene", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    asset = load_robot_asset(args.asset)
    probs = load_vamp_problems(args.problems, max_per_scene=args.max_per_scene)

    w = []
    def emit(name, values):
        vals = np.asarray(values).ravel()
        w.append(f"{name} {vals.size}")
        w.append(" ".join(repr(float(v)) if vals.dtype.kind == "f" else str(int(v))
                          for v in vals))

    w.append(f"n_dof {asset['n_dof']}")
    w.append(f"n_joints {asset['n_joints']}")
    w.append(f"n_spheres {len(asset['sphere_to_joint'])}")
    emit("fixed_transforms", np.asarray(asset["fixed_transforms"], float))
    emit("joint_types", np.asarray(asset["joint_types"], int))
    emit("joint_parents", np.asarray(asset["joint_parents"], int))
    emit("joint_id_to_dof", np.asarray(asset["joint_id_to_dof"], int))
    emit("dfs_order", np.asarray(asset["dfs_order"], int))
    emit("spheres", np.asarray(asset["spheres_flat"], float))
    emit("sphere_to_joint", np.asarray(asset["sphere_to_joint"], int))
    emit("joint_lower", np.asarray(asset["joint_lower"], float))
    emit("joint_upper", np.asarray(asset["joint_upper"], float))

    w.append(f"n_problems {len(probs)}")
    for p in probs:
        obbs = obstacles_to_obbs(p.obstacles)
        w.append(f"problem {p.scene} {p.index} {len(obbs)}")
        emit("start", np.asarray(p.start, float))
        emit("goal", np.asarray(p.goal, float))
        for d in obbs:
            q = np.asarray(d.get("quaternion", (1.0, 0.0, 0.0, 0.0)), float)
            n = np.linalg.norm(q)
            if n > 0:
                q = q / n
            emit("obb", np.concatenate([np.asarray(d["dims"], float),
                                        np.asarray(d["position"], float), q]))

    Path(args.out).write_text("\n".join(w) + "\n")
    print(f"wrote {args.out}: {asset['name']} n_dof={asset['n_dof']} "
          f"problems={len(probs)}")


if __name__ == "__main__":
    main()
