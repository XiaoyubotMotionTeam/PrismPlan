"""Cross-paradigm run with an EQUAL time budget for every planner.

This is the driver behind the cross-paradigm table and figures: it is the only
entry point that gives every planner the same wall-clock budget, which
``bench_mbm.py`` and ``bench_6way.py`` do not.

Why a separate driver. ``bench_mbm.py`` instantiates each planner with its
shipped YAML, which hands pRRTC 4000 ms and the search/MIT* planners 1000 ms --
not comparable. Here every planner gets the same ``--budget-ms``, STOMP and
CHOMP included: their solve loops DO honour ``time_limit_ms`` (inherited from
``pRRTC_settings``), they were merely capped by ``num_iterations`` long before
it and silently inherited the 4000 ms default when it was left unset.

An equal *nominal* budget is not an equal *effective* budget. Every planner in
SPECS was audited for non-time termination conditions before the numbers in the
paper were produced, because several had one:

  STOMP   num_iterations 20     -> loop ended at ~12.6 ms, ~1 % of a 1 s budget
  CHOMP   num_iterations 100    -> same class of cap
  MIT*    max_nodes 50000       -> GPU buffer bound; a nominal 2 s Panda run
                                   actually stopped at a 1211 ms median
  pRRTC   max_samples/max_iters -> measured NOT binding at 2 s (identical
                                   success at 1e6 and 4e6); left alone, since
                                   raising it only adds allocation overhead
  MHA*    mhastar_search.hh     -> time-bound only (verified by reading the loop)
  wPA*SE  wpase_search.hh       -> time-bound only (verified by reading the loop)

Before quoting any "equal budget" result, re-run that audit: grep every
termination condition in the solve loop, not just the time one.

Problem source (--source): the paper cites the Panda set "as distributed by
robometrics", which is 800 problems over 8 scene families. The MBM MoveIt
tarball converted by ``tools/mbm_tar_to_json.py`` holds only 7 of those families
-- upstream MotionBenchMaker has no box_panda_flipped scene -- so it is a
700-problem set and measurably easier (pRRTC 699/700 there vs 785/800 on the
robometrics set). Use ``--source robometrics`` for anything that goes into the
paper.

Usage:
    python3 examples/bench_all.py --asset config/robots/panda.yaml \
        --source robometrics --max-per-scene 100 --budget-ms 2000
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from gpu_planning.benchmark import load_vamp_problems, run_benchmark  # noqa: E402
from gpu_planning.benchmark.loaders import load_robometrics_problems  # noqa: E402
from gpu_planning.plugins._substrate_planner_base import _flatten_config  # noqa: E402
from gpu_planning.plugins.chomp_plugin import CHOMPPlanner  # noqa: E402
from gpu_planning.plugins.mhastar_plugin import MHAStarPlanner  # noqa: E402
from gpu_planning.plugins.mitstar_plugin import MITStarPlanner  # noqa: E402
from gpu_planning.plugins.prrtc_plugin import PRRTCPlanner  # noqa: E402
from gpu_planning.plugins.stomp_plugin import STOMPPlanner  # noqa: E402
from gpu_planning.plugins.wpase_plugin import WPASEPlanner  # noqa: E402

# label -> (class, yaml, config-key prefix for the time limit)
SPECS = [
    ("pRRTC",  PRRTCPlanner,   "prrtc.yaml",   ""),
    ("MIT*",   MITStarPlanner, "mitstar.yaml", "mitstar_"),
    ("MHA*",   MHAStarPlanner, "mhastar.yaml", "mhastar_"),
    ("wPA*SE", WPASEPlanner,   "wpase.yaml",   "wpase_"),
    ("STOMP",  STOMPPlanner,   "stomp.yaml",   "stomp_"),
    ("CHOMP",  CHOMPPlanner,   "chomp.yaml",   "chomp_"),
]


def build(budget_ms, extra):
    out = []
    for label, cls, fname, prefix in SPECS:
        raw = yaml.safe_load((ROOT / "config" / "planners" / fname).read_text())
        cfg = _flatten_config(raw)
        if prefix is not None:
            # Set both the bare and prefixed key: MHA*/wPA*SE carry a base
            # pRRTC section whose unprefixed time_limit_ms also gates them.
            cfg["time_limit_ms"] = float(budget_ms)
            if prefix:
                cfg[f"{prefix}time_limit_ms"] = float(budget_ms)
        cfg.update(extra)
        out.append((label, partial(cls, config=cfg)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--path", default=None,
                    help="VAMP-format json/pkl (required for --source vamp)")
    ap.add_argument("--source", choices=("vamp", "robometrics"), default="vamp",
                    help="problem source. 'robometrics' reads the datasets the "
                         "robometrics package ships, which is what the paper "
                         "cites; the MBM tarball conversion under --source vamp "
                         "is missing the box_panda_flipped family (700 vs 800).")
    ap.add_argument("--dataset", default="mbm",
                    help="robometrics dataset name (mbm | mpinets | demo)")
    ap.add_argument("--max-per-scene", type=int, default=10)
    ap.add_argument("--budget-ms", type=float, default=1000.0)
    ap.add_argument("--planners", default=None)
    ap.add_argument("--out", default=None,
                    help="directory for trials.csv + summary.json. Required if "
                         "the run feeds the paper figures: examples/plot_mbm.py "
                         "reads both, and neither can be reconstructed from "
                         "stdout without inventing per-trial rows.")
    ap.add_argument("--set", action="append", default=[],
                    help="extra flat config override, k=v")
    args = ap.parse_args()

    extra = {}
    for item in args.set:
        k, _, v = item.partition("=")
        extra[k.strip()] = yaml.safe_load(v)

    if args.source == "robometrics":
        probs, skipped = load_robometrics_problems(
            args.dataset, max_per_scene=args.max_per_scene)
        # Loudly: a nonzero skip count silently shrinks the denominator, which
        # is exactly the kind of thing that must not reach the paper unnoticed.
        print(f"source=robometrics dataset={args.dataset} skipped={skipped}")
    else:
        if not args.path:
            ap.error("--path is required with --source vamp")
        probs = load_vamp_problems(args.path, max_per_scene=args.max_per_scene)
    specs = build(args.budget_ms, extra)
    if args.planners:
        want = [s.strip() for s in args.planners.split(",")]
        specs = [s for s in specs if s[0] in want]

    n_scenes = len({p.scene for p in probs})
    print(f"asset={args.asset} source={args.source} problems={len(probs)} "
          f"scenes={n_scenes} budget={args.budget_ms:.0f}ms "
          f"extra={extra or '{}'}\n")

    _, summary = run_benchmark(specs, probs, asset_path=args.asset, device="cuda",
                               trials=1, out_dir=args.out)

    hdr = (f"{'planner':8s} {'succ':>8s} {'rate':>6s} {'mean_ms':>9s} "
           f"{'median_ms':>10s} {'cost':>9s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, block in summary["by_planner"].items():
        a = block["all"]
        print(f"{label:8s} {a['successes']:>4d}/{a['attempts']:<3d} "
              f"{a['success_rate']:>6.2f} {a['solve_ms']['mean']:>9.1f} "
              f"{a['solve_ms']['median']:>10.1f} {a['cost']['mean']:>9.4g}")

    print("\nper-scene success rate")
    for label, block in summary["by_planner"].items():
        parts = [f"{scene}={s['successes']}/{s['attempts']}"
                 for scene, s in sorted(block.get("by_scene", {}).items())]
        print(f"  {label:8s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
