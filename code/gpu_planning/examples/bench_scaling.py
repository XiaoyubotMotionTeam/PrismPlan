# -*- coding: utf-8 -*-
"""bench_scaling — pRRTC GPU parallel-width (num_new_configs) sweep.

This measures how the sampling planner scales with the *one* runtime GPU-
parallelism knob the shipped kernel actually exposes: ``num_new_configs``, the
number of parallel tree-extension blocks (the CUDA launch is
``rrtc_runtime<<<num_new_configs, 4*granularity>>>``). For each width we plan a
**fixed** set of MotionBenchMaker problems on the shared substrate and record:

* median / mean solve time on solved problems,
* success rate (a too-narrow tree may fail within the budget),
* peak device memory *used* during the sweep.

Memory is read via ``torch.cuda.mem_get_info()`` (driver free/total), so it
captures ``prrtc``'s own ``cudaMalloc`` buffers — ``torch.cuda.max_memory_*``
would miss them because prrtc does not allocate through torch's caching
allocator. We report the delta above a per-process baseline as the planner's
device footprint.

This is deliberately a single-planner, single-knob scaling study: the package
has **no** concurrent-batch (solve-B-problems-at-once) API — every
``*_solve_scene`` solves one (start, goals) problem — so "throughput vs. batch
size B" is not something the shipped code can produce. What it *can* show
honestly is latency / success / memory vs. intra-solve parallel width.

Usage::

    python examples/bench_scaling.py --dataset mbm \\
        --scenes box_panda,table_pick_panda,cage_panda --per-scene 8 \\
        --widths 1,2,4,8,16,32,64,128,256,600,1024 \\
        --time-limit-ms 1000 --out results/scaling
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Allow running directly without installing: add the package's python/ dir.
_PKG_PY = Path(__file__).resolve().parents[1] / "python"
if _PKG_PY.is_dir() and str(_PKG_PY) not in sys.path:
    sys.path.insert(0, str(_PKG_PY))

from gpu_planning import PRRTCPlanner  # noqa: E402
from gpu_planning.base.planner_base import PlanningRequest  # noqa: E402
from gpu_planning.benchmark import load_robometrics_problems  # noqa: E402
from gpu_planning.benchmark.problem import obstacles_to_obbs  # noqa: E402
from gpu_planning.substrate import PlanningScene, load_robot_asset  # noqa: E402

logger = logging.getLogger("bench_scaling")


def _mem_used_mb(device_index: int = 0) -> float:
    """Device memory currently in use (MiB), read from the CUDA driver.

    Uses driver free/total so it reflects ALL allocations on the device,
    including prrtc's raw cudaMalloc buffers (which torch's allocator stats
    never see).
    """
    import torch
    free, total = torch.cuda.mem_get_info(device_index)
    return (total - free) / (1024.0 * 1024.0)


def _stats(xs):
    if not xs:
        return {"mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(statistics.fmean(xs)),
        "median": float(statistics.median(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def _run_point(knob, value, problems, scene, robot_id, asset_path,
               time_limit_ms, csv_rows):
    """Plan every problem at one knob value; return the summary row.

    ``knob`` is the pRRTC config key to sweep (``num_new_configs``,
    ``max_samples``, or ``granularity``). A fresh pre-init baseline is taken per
    point so the reported planner footprint isolates pRRTC's own allocation even
    if a co-resident GPU process drifts between points.
    """
    pre_mb = _mem_used_mb()  # immediately before this point's planner exists
    planner = PRRTCPlanner()
    # Override the swept knob + cap the per-solve budget so failures don't
    # dominate the wall clock. All three sweepable keys are read by
    # _init_buffers (buffer sizing) and/or _build_settings (kernel launch), so
    # setting them before initialize() is sufficient.
    planner._config[knob] = int(value)
    planner._config["time_limit_ms"] = int(time_limit_ms)

    if not planner.initialize(scene, asset_path=asset_path):
        raise RuntimeError(f"pRRTC failed to initialize at {knob}={value}")

    peak_mb = _mem_used_mb()  # after buffers allocated
    solve_ms_ok, costs_ok = [], []
    n_ok = 0
    try:
        for prob in problems:
            scene.update_world(obstacles_to_obbs(prob.obstacles))
            request = PlanningRequest(
                robot_id=robot_id,
                start_joint_state=np.asarray(prob.start, dtype=np.float32),
                target_joint_state=np.asarray(prob.goal, dtype=np.float32),
                velocity_scaling=1.0, acceleration_scaling=1.0)
            t0 = time.time()
            result = planner.plan(request)
            wall_ms = (time.time() - t0) * 1e3
            info = result.extra_info or {}
            solver_ms = float(info.get("wall_time_ms", wall_ms))
            ok = bool(result.success)
            if ok:
                n_ok += 1
                solve_ms_ok.append(solver_ms)
                if "cost" in info:
                    try:
                        costs_ok.append(float(info["cost"]))
                    except (TypeError, ValueError):
                        pass
            peak_mb = max(peak_mb, _mem_used_mb())
            csv_rows.append({
                "knob": knob, "value": int(value), "scene": prob.scene,
                "problem_index": prob.index, "success": ok,
                "status": result.status or "",
                "solve_ms": round(solver_ms, 4),
                "cost": info.get("cost", ""),
            })
    finally:
        planner.shutdown()

    n = len(problems)
    st = _stats(solve_ms_ok)
    row = {
        "knob": knob,
        "value": int(value),
        "attempts": n,
        "successes": n_ok,
        "success_rate": (n_ok / n) if n else float("nan"),
        "solve_ms": st,
        "cost": _stats(costs_ok),
        "peak_mem_mb": round(peak_mb, 1),
        "planner_mem_mb": round(peak_mb - pre_mb, 1),
    }
    print(f"  {knob}={value:>8d}  succ={n_ok:>3d}/{n:<3d} "
          f"median_ms={st['median']:>8.1f}  mean_ms={st['mean']:>8.1f}  "
          f"planner_mem=+{peak_mb - pre_mb:>7.1f}MiB")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mbm", help="robometrics dataset")
    ap.add_argument("--scenes", default="box_panda,table_pick_panda,cage_panda",
                    help="comma-separated scene subset")
    ap.add_argument("--per-scene", type=int, default=8,
                    help="problems per scene (fixed across all sweep points)")
    ap.add_argument("--knob", default="num_new_configs",
                    choices=("num_new_configs", "max_samples", "granularity"),
                    help="pRRTC config key to sweep")
    ap.add_argument("--values",
                    default="1,2,4,8,16,32,64,128,256,600,1024",
                    help="comma-separated values of --knob to sweep")
    ap.add_argument("--time-limit-ms", type=int, default=1000)
    ap.add_argument("--asset", default=None, help="robot YAML (default panda)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR,
                        format="%(message)s")

    try:
        import prrtc  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        print(f"ERROR: need compiled prrtc + torch on a CUDA device: {e}",
              file=sys.stderr)
        return 1

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    values = [int(v) for v in args.values.split(",") if v.strip()]

    problems, skipped = load_robometrics_problems(
        args.dataset, scenes=scenes, max_per_scene=args.per_scene)
    if any(skipped.values()):
        print(f"skipped: {skipped}", file=sys.stderr)
    if not problems:
        print("ERROR: no usable problems loaded", file=sys.stderr)
        return 3
    print(f"loaded {len(problems)} problems across {len(scenes)} scenes; "
          f"knob={args.knob} values={values}")

    asset = load_robot_asset(args.asset)
    robot_id = asset["name"]
    scene = PlanningScene.from_robot_asset(args.asset, device=args.device)

    baseline_mb = _mem_used_mb()
    print(f"device memory in use before planners (co-resident baseline): "
          f"{baseline_mb:.1f} MiB")

    _KNOB_NOTE = {
        "num_new_configs": (
            "num_new_configs = parallel tree-extension block count; launch is "
            "rrtc_runtime<<<num_new_configs,4*granularity>>>. No concurrent-batch "
            "(multi-problem) API exists; this is intra-solve parallel width, not "
            "batch size B."),
        "max_samples": (
            "max_samples = per-tree node capacity; drives the dominant device "
            "buffer. Sweep reveals the memory ceiling vs. tree capacity."),
        "granularity": (
            "granularity = waypoints checked per edge (<= BATCH_SIZE 16); the "
            "only runtime-toggleable backend knob. The other named backend "
            "factors (two-phase early-exit, SoA layout, 4-thread cooperative "
            "check) are compile-time constants, not runtime-ablatable."),
    }

    csv_rows = []
    rows = []
    for v in values:
        rows.append(_run_point(
            args.knob, v, problems, scene, robot_id, args.asset,
            args.time_limit_ms, csv_rows))

    summary = {
        "config": {
            "robot": robot_id, "device": args.device, "dataset": args.dataset,
            "scenes": scenes, "per_scene": args.per_scene,
            "n_problems": len(problems), "knob": args.knob, "values": values,
            "time_limit_ms": args.time_limit_ms,
            "baseline_mem_mb": round(baseline_mb, 1),
            "note": _KNOB_NOTE[args.knob],
        },
        "by_value": rows,
    }

    if args.out:
        d = Path(args.out)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "scaling_trials.csv").open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(csv_rows)
        with (d / "scaling_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {d/'scaling_trials.csv'} and scaling_summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
