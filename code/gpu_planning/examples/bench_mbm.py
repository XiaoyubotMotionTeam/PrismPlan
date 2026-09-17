# -*- coding: utf-8 -*-
"""bench_mbm — run the substrate planners over MotionBenchMaker problems.

Loads a MotionBenchMaker (MBM) problem set from one of two sources, normalises
its box/sphere/cylinder geometry into conservative OBBs, and benchmarks every
substrate-consistent planner over it on the shared GPU collision core.

Sources:

  * ``--source vamp --path problems.pkl``  — the ``.pkl``/``.json`` produced by
    VAMP's ``problem_tar_to_pkl_json.py`` (joint-space goals; all usable).
  * ``--source robometrics --dataset mbm`` — the ``robometrics`` datasets
    (``mbm`` / ``mpinets`` / ``demo``). Goals are end-effector poses; problems
    without a joint-space IK goal are skipped (this package ships no IK).

Requires the compiled ``prrtc`` pybind module (and a CUDA device) on the
``PYTHONPATH`` to actually plan. A tiny synthetic VAMP-format sample ships at
``examples/data/sample_mbm_problems.json`` for wiring checks:

    python examples/bench_mbm.py --source vamp \\
        --path examples/data/sample_mbm_problems.json --trials 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running directly without installing: add the package's python/ dir.
_PKG_PY = Path(__file__).resolve().parents[1] / "python"
if _PKG_PY.is_dir() and str(_PKG_PY) not in sys.path:
    sys.path.insert(0, str(_PKG_PY))

from gpu_planning import (  # noqa: E402
    CHOMPPlanner,
    MHAStarPlanner,
    MITStarPlanner,
    PRRTCPlanner,
    STOMPPlanner,
    WPASEPlanner,
)
from gpu_planning.benchmark import (  # noqa: E402
    load_robometrics_problems,
    load_vamp_problems,
    run_benchmark,
)

ALL_PLANNERS = {
    "pRRTC": PRRTCPlanner,
    "MIT*": MITStarPlanner,
    "MHA*": MHAStarPlanner,
    "wPA*SE": WPASEPlanner,
    "STOMP": STOMPPlanner,
    "CHOMP": CHOMPPlanner,
}


def _print_summary(summary):
    cfg = summary.get("config", {})
    print(f"\nrobot={cfg.get('robot')} device={cfg.get('device')} "
          f"problems={cfg.get('n_problems')} trials={cfg.get('trials')}")
    header = (f"{'planner':8s} {'succ':>6s} {'rate':>6s} "
              f"{'solve_ms(mean)':>14s} {'median':>8s} {'cost(mean)':>11s}")
    print(header)
    print("-" * len(header))
    for planner, block in summary["by_planner"].items():
        a = block["all"]
        print(f"{planner:8s} "
              f"{a['successes']:>3d}/{a['attempts']:<2d} "
              f"{a['success_rate']:>6.2f} "
              f"{a['solve_ms']['mean']:>14.1f} "
              f"{a['solve_ms']['median']:>8.1f} "
              f"{a['cost']['mean']:>11.4g}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("vamp", "robometrics"),
                        default="vamp", help="benchmark data source")
    parser.add_argument("--path", default=None,
                        help="VAMP .pkl/.json path (--source vamp)")
    parser.add_argument("--dataset", default="mbm",
                        help="robometrics dataset: mbm|mpinets|demo")
    parser.add_argument("--asset", default=None,
                        help="robot YAML asset (default: bundled panda)")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--planners", default=None,
                        help="comma-separated subset (default: all six)")
    parser.add_argument("--scenes", default=None,
                        help="comma-separated scene subset (default: all)")
    parser.add_argument("--trials", type=int, default=1,
                        help="repeats per planner×problem")
    parser.add_argument("--max-per-scene", type=int, default=None,
                        help="cap problems loaded per scene")
    parser.add_argument("--budget-ms", type=float, default=None,
                        help="equal wall-clock budget in ms for EVERY planner; "
                             "omit to use each planner's own YAML default "
                             "(those defaults are not equal across planners)")
    parser.add_argument("--out", default=None,
                        help="output dir for trials.csv + summary.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s")

    try:
        import prrtc  # noqa: F401
    except ImportError:
        print("ERROR: the compiled `prrtc` module is not importable — build "
              "the pybind extension in cpp/ first (needs a CUDA toolchain).",
              file=sys.stderr)
        return 1

    scenes = args.scenes.split(",") if args.scenes else None

    if args.source == "vamp":
        if not args.path:
            print("ERROR: --source vamp requires --path", file=sys.stderr)
            return 2
        problems = load_vamp_problems(
            args.path, scenes=scenes, max_per_scene=args.max_per_scene)
    else:
        problems, skipped = load_robometrics_problems(
            args.dataset, scenes=scenes, max_per_scene=args.max_per_scene)
        if any(skipped.values()):
            print(f"skipped: {skipped}", file=sys.stderr)

    if not problems:
        print("ERROR: no usable problems loaded", file=sys.stderr)
        return 3
    print(f"loaded {len(problems)} problems from {args.source}")

    if args.planners:
        want = [p.strip() for p in args.planners.split(",")]
        specs = [(lbl, ALL_PLANNERS[lbl]) for lbl in want if lbl in ALL_PLANNERS]
    else:
        specs = list(ALL_PLANNERS.items())

    if args.budget_ms is not None:
        print(f"equal wall-clock budget: {args.budget_ms:.0f} ms per planner")

    _, summary = run_benchmark(
        specs, problems, asset_path=args.asset, device=args.device,
        trials=args.trials, out_dir=args.out, budget_ms=args.budget_ms)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
