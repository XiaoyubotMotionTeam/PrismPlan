"""Sweep MIT* settings over a problem set and report success rate per variant.

Exists so MIT* tuning is measured against the CPU OMPL MITstar baseline
(tools/ompl_mitstar_bench.cpp) rather than guessed. Variants are given as
`key=value` overrides of the `mitstar_*` config keys, e.g.

    python3 tools/mitstar_sweep.py --path /tmp/panda_problems.json \
        --asset config/robots/panda.yaml --max-per-scene 10 \
        --variant baseline \
        --variant 'bs256:batch_size=256' \
        --variant 'more_fwd:m_forward_eval=1024,m_reverse_eval=256'
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_PKG_PY = Path(__file__).resolve().parents[1] / "python"
if _PKG_PY.is_dir() and str(_PKG_PY) not in sys.path:
    sys.path.insert(0, str(_PKG_PY))

from gpu_planning.base.planner_base import PlanningRequest  # noqa: E402
from gpu_planning.benchmark import load_vamp_problems, obstacles_to_obbs  # noqa: E402
from gpu_planning.plugins.mitstar_plugin import MITStarPlanner  # noqa: E402
from gpu_planning.substrate import PlanningScene, load_robot_asset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def parse_variant(spec):
    """'name:k=v,k=v' -> (name, {mitstar_k: v}). Bare 'name' means no override."""
    name, _, body = spec.partition(":")
    over = {}
    for item in filter(None, body.split(",")):
        k, _, v = item.partition("=")
        try:
            val = yaml.safe_load(v)
        except yaml.YAMLError:
            val = v
        over[f"mitstar_{k.strip()}"] = val
    return name, over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--max-per-scene", type=int, default=None)
    ap.add_argument("--scenes", default=None, help="comma-separated substrings")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--variant", action="append", default=[])
    args = ap.parse_args()

    probs = load_vamp_problems(args.path, max_per_scene=args.max_per_scene)
    if args.scenes:
        keys = [s for s in args.scenes.split(",") if s]
        probs = [p for p in probs if any(k in p.scene for k in keys)]

    base_cfg = yaml.safe_load((ROOT / "config" / "planners" / "mitstar.yaml")
                              .read_text())
    asset = load_robot_asset(args.asset)
    scene = PlanningScene.from_robot_asset(args.asset, device="cuda")

    variants = [parse_variant(v) for v in (args.variant or ["baseline"])]
    print(f"{len(probs)} problems x {args.trials} trials x "
          f"{len(variants)} variants\n")

    for name, over in variants:
        cfg = dict(base_cfg)
        cfg.update(over)
        planner = MITStarPlanner(config=cfg)
        if not planner.initialize(scene, asset_path=args.asset):
            print(f"{name}: initialize failed")
            continue

        by_scene = defaultdict(lambda: [0, 0])
        ms_ok = []
        try:
            for p in probs:
                scene.update_world(obstacles_to_obbs(p.obstacles))
                req = PlanningRequest(
                    robot_id=asset["name"],
                    start_joint_state=np.asarray(p.start, np.float32),
                    target_joint_state=np.asarray(p.goal, np.float32),
                    velocity_scaling=1.0, acceleration_scaling=1.0)
                for _ in range(args.trials):
                    t0 = time.time()
                    res = planner.plan(req)
                    ms = (time.time() - t0) * 1e3
                    agg = by_scene[p.scene]
                    agg[1] += 1
                    if res.success:
                        agg[0] += 1
                        ms_ok.append(ms)
        finally:
            planner.shutdown()

        tot_ok = sum(a[0] for a in by_scene.values())
        tot_n = sum(a[1] for a in by_scene.values())
        print(f"=== {name}  {over if over else '(defaults)'}")
        for s in sorted(by_scene):
            ok, n = by_scene[s]
            print(f"  {s:<34} {ok:3d}/{n:<3d} {ok / n:.2f}")
        print(f"  {'TOTAL':<34} {tot_ok:3d}/{tot_n:<3d} {tot_ok / tot_n:.2f}"
              f"   mean solve {np.mean(ms_ok) if ms_ok else float('nan'):.1f} ms\n")


if __name__ == "__main__":
    main()
