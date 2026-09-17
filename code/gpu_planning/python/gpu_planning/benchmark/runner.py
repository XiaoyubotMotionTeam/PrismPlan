# -*- coding: utf-8 -*-
"""Benchmark run loop + metric aggregation.

Runs a set of planners over a set of normalized :class:`BenchProblem`s on the
**shared** GPU collision substrate and collects comparable metrics. The loop is
structured problem-outer / planner-inner: the obstacle world is pushed to a
single :class:`~gpu_planning.PlanningScene` once per problem, so every planner
plans against a byte-identical world — the internal-comparability property the
substrate-consistency thesis relies on.

Results are written as a per-trial CSV (one row per planner×problem×trial) and a
summary JSON (aggregates per planner and per planner×scene). ``prrtc`` is never
imported at module load — the planner subclasses import it lazily inside
``initialize`` / ``plan``.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from ..base.planner_base import PlannerBase, PlanningRequest, PlanningResult
from ..substrate import PlanningScene, load_robot_asset
from .problem import BenchProblem, obstacles_to_obbs

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Per-trial record
# --------------------------------------------------------------------------- #
@dataclass
class TrialRecord:
    planner: str
    scene: str
    problem_index: int
    trial: int
    success: bool
    status: str
    solve_ms: float
    gpu_ms: float
    waypoints: int
    total_time: float
    cost: float
    expansions: int
    edges_evaluated: int
    error: str = ""


def _cost_of(result: PlanningResult) -> float:
    info = result.extra_info or {}
    for key in ("cost", "final_total_cost"):
        if key in info:
            try:
                return float(info[key])
            except (TypeError, ValueError):
                pass
    return float("nan")


def _int_of(info: Dict[str, Any], key: str) -> int:
    try:
        return int(info.get(key, -1))
    except (TypeError, ValueError):
        return -1


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _summarize(records: Sequence[TrialRecord]) -> Dict[str, Any]:
    """Aggregate stats over a homogeneous slice of trial records."""
    n = len(records)
    ok = [r for r in records if r.success]
    solve_ok = [r.solve_ms for r in ok]
    cost_ok = [r.cost for r in ok if not np.isnan(r.cost)]

    def _stats(xs: Sequence[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": float("nan"), "std": float("nan"),
                    "median": float("nan"), "min": float("nan"),
                    "max": float("nan")}
        return {
            "mean": float(statistics.fmean(xs)),
            "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
            "median": float(statistics.median(xs)),
            "min": float(min(xs)),
            "max": float(max(xs)),
        }

    return {
        "attempts": n,
        "successes": len(ok),
        "success_rate": (len(ok) / n) if n else float("nan"),
        "solve_ms": _stats(solve_ok),
        "cost": _stats(cost_ok),
    }


def aggregate(records: Sequence[TrialRecord]) -> Dict[str, Any]:
    """Build the nested summary: overall, per-planner, per planner×scene."""
    planners = sorted({r.planner for r in records})
    scenes = sorted({r.scene for r in records})
    summary: Dict[str, Any] = {
        "overall": _summarize(records),
        "by_planner": {},
    }
    for p in planners:
        p_recs = [r for r in records if r.planner == p]
        summary["by_planner"][p] = {
            "all": _summarize(p_recs),
            "by_scene": {
                s: _summarize([r for r in p_recs if r.scene == s])
                for s in scenes if any(r.scene == s for r in p_recs)
            },
        }
    return summary


# --------------------------------------------------------------------------- #
# Run loop
# --------------------------------------------------------------------------- #
def run_benchmark(
    planner_specs: Sequence[Tuple[str, Type[PlannerBase]]],
    problems: Sequence[BenchProblem],
    asset_path: Optional[str] = None,
    device: str = "cuda",
    trials: int = 1,
    out_dir: Optional[str] = None,
    budget_ms: Optional[float] = None,
) -> Tuple[List[TrialRecord], Dict[str, Any]]:
    """Plan every ``problem`` with every planner ``trials`` times.

    One :class:`PlanningScene` and one planner instance per planner class are
    created up front and reused; the world is refreshed per problem via
    ``update_world``. Returns ``(records, summary)`` and, if ``out_dir`` is
    given, also writes ``trials.csv`` and ``summary.json`` there.

    ``budget_ms`` overrides every planner's wall-clock budget with the same
    value, which is what an equal-budget comparison needs; leave it ``None`` to
    use each planner's own YAML default (those defaults are *not* equal across
    planners). The value used is recorded in ``summary["config"]["budget_ms"]``
    so an artifact always says which budget produced it.
    """
    asset = load_robot_asset(asset_path)
    robot_id = asset["name"]

    scene = PlanningScene.from_robot_asset(asset_path, device=device)

    # Initialize each planner once against the shared scene.
    planners: List[Tuple[str, PlannerBase]] = []
    for label, cls in planner_specs:
        inst = cls()
        if budget_ms is not None:
            setter = getattr(inst, "set_time_budget_ms", None)
            if setter is None:
                raise TypeError(
                    f"--budget-ms given but {label} ({type(inst).__name__}) has "
                    "no set_time_budget_ms; it would silently run at its own "
                    "default budget")
            setter(budget_ms)
        if not inst.initialize(scene, asset_path=asset_path):
            logger.warning("[bench] %s failed to initialize; skipping", label)
            continue
        planners.append((label, inst))

    records: List[TrialRecord] = []
    try:
        for prob in problems:
            scene.update_world(obstacles_to_obbs(prob.obstacles))
            request = PlanningRequest(
                robot_id=robot_id,
                start_joint_state=np.asarray(prob.start, dtype=np.float32),
                target_joint_state=np.asarray(prob.goal, dtype=np.float32),
                velocity_scaling=1.0, acceleration_scaling=1.0)

            for label, planner in planners:
                for t in range(trials):
                    rec = _run_one(label, planner, request, prob, t)
                    records.append(rec)
                    logger.info(
                        "[bench] %-8s %s#%d trial %d -> %s %.1fms",
                        label, prob.scene, prob.index, t, rec.status,
                        rec.solve_ms)
    finally:
        for _, planner in planners:
            try:
                planner.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("[bench] shutdown failed")

    summary = aggregate(records)
    summary["config"] = {
        "robot": robot_id, "device": device, "trials": trials,
        "n_problems": len(problems),
        "planners": [lbl for lbl, _ in planner_specs],
        "budget_ms": budget_ms,
    }

    if out_dir is not None:
        _write_outputs(records, summary, out_dir)

    return records, summary


def _run_one(label: str, planner: PlannerBase, request: PlanningRequest,
             prob: BenchProblem, trial: int) -> TrialRecord:
    start = time.time()
    try:
        result = planner.plan(request)
    except Exception as e:  # noqa: BLE001
        logger.exception("[bench] %s raised on %s#%d", label, prob.scene, prob.index)
        return TrialRecord(
            planner=label, scene=prob.scene, problem_index=prob.index,
            trial=trial, success=False, status="Exception",
            solve_ms=(time.time() - start) * 1e3, gpu_ms=float("nan"),
            waypoints=0, total_time=0.0, cost=float("nan"),
            expansions=-1, edges_evaluated=-1, error=str(e))

    info = result.extra_info or {}
    return TrialRecord(
        planner=label, scene=prob.scene, problem_index=prob.index, trial=trial,
        success=bool(result.success), status=result.status or "",
        solve_ms=float(result.solve_time) * 1e3,
        gpu_ms=float(info.get("wall_time_ms", float("nan"))),
        waypoints=int(info.get("geometric_waypoints", result.num_waypoints)),
        total_time=float(result.total_time),
        cost=_cost_of(result),
        expansions=_int_of(info, "expansions"),
        edges_evaluated=_int_of(info, "edges_evaluated"),
        error=result.error_message or "")


def _write_outputs(records: Sequence[TrialRecord], summary: Dict[str, Any],
                   out_dir: str) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    csv_path = d / "trials.csv"
    fields = list(asdict(records[0]).keys()) if records else [
        f.name for f in TrialRecord.__dataclass_fields__.values()]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    with (d / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[bench] wrote %s and summary.json", csv_path)
