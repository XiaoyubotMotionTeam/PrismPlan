# -*- coding: utf-8 -*-
"""wPA*SE — weighted Parallel A* for Slow Expansions (search paradigm).

CPU parallel-expansion weighted-A* over a joint lattice: each round expands a
set of mutually-independent states (PA*SE independence rule) whose successor
edges are all confirmed on the GPU in one batch against the shared collision
substrate. Suboptimality is bounded by the heuristic weight ``heuristic_w``.
Joint-space start->goal only. No reusable GPU buffers — the solve entry point
manages its own edge-evaluation scratch.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..base.planner_base import PlannerType, PlanningRequest, PlanningResult
from ._substrate_planner_base import SubstratePlannerBase

logger = logging.getLogger(__name__)


class WPASEPlanner(SubstratePlannerBase):
    """weighted PA*SE parallel search with GPU-batched edge evaluation."""

    _DEFAULT_YAML = "config/planners/wpase.yaml"

    def __init__(self, config=None, config_path=None):
        super().__init__(PlannerType.WPASE, config=config,
                         config_path=config_path)

    def _build_settings(self):
        import prrtc
        settings = prrtc.WPASESettings()
        for key, val in self._config.items():
            if key.startswith("wpase_"):
                attr = key[len("wpase_"):]
                if hasattr(settings, attr):
                    setattr(settings, attr, val)
        if "collision_margin" in self._config:
            settings.collision_margin = self._config["collision_margin"]
        return settings

    def plan(self, request: PlanningRequest) -> PlanningResult:
        err = self._guard_request(request)
        if err is not None:
            return err

        import prrtc
        start_time = time.time()
        start = np.asarray(request.start_joint_state, dtype=np.float32)
        goal = np.asarray(request.target_joint_state, dtype=np.float32)
        settings = self._build_settings()

        try:
            with self._solve_context() as scene_coll:
                if scene_coll is None:
                    return PlanningResult(
                        success=False, status="PlanningError",
                        error_message="wPA*SE requires scene collision data "
                                      "(the world has no obstacles)",
                        solve_time=time.time() - start_time)

                # The lattice search cannot legally leave a colliding start or
                # enter a colliding goal — both are fatal.
                flags = self._preflight(scene_coll, np.vstack([start, goal]))
                if bool(flags[0]):
                    return PlanningResult(
                        success=False, status="StartInCollision",
                        error_message="wPA*SE start configuration in collision",
                        solve_time=time.time() - start_time)
                if bool(flags[1]):
                    return PlanningResult(
                        success=False, status="GoalInCollision",
                        error_message="wPA*SE goal configuration in collision",
                        solve_time=time.time() - start_time)

                result = prrtc.wpase_solve_scene(
                    start, np.asarray([goal], dtype=np.float32),
                    scene_coll, settings, self._robot_model)
        except Exception as e:  # noqa: BLE001
            logger.exception("[wPA*SE] solver raised")
            return PlanningResult(
                success=False, status="PlanningError",
                error_message=f"wPA*SE planning failed: {e}",
                solve_time=time.time() - start_time)

        solve_time = time.time() - start_time
        if not result.solved:
            logger.warning("[wPA*SE] no solution: expansions=%s edges=%s wall=%.1fms",
                           result.expansions, result.edges_evaluated,
                           result.wall_time_ms)
            return PlanningResult(
                success=False, status="PlanningFailed",
                error_message="wPA*SE did not find a path within the "
                              f"time/expansion budget (expansions={result.expansions})",
                solve_time=solve_time,
                extra_info={"solver": "wpase", "expansions": result.expansions,
                            "edges_evaluated": result.edges_evaluated,
                            "wall_time_ms": result.wall_time_ms})

        path = np.asarray(result.path, dtype=np.float32)
        logger.info("[wPA*SE] solved: expansions=%s edges=%s cost=%.4g waypoints=%d",
                    result.expansions, result.edges_evaluated, result.cost,
                    len(path))
        return self._finalize(
            path, request, solve_time,
            extra_info={"solver": "wpase", "expansions": result.expansions,
                        "edges_evaluated": result.edges_evaluated,
                        "cost": result.cost, "wall_time_ms": result.wall_time_ms,
                        "geometric_path": path, "geometric_waypoints": len(path)})
