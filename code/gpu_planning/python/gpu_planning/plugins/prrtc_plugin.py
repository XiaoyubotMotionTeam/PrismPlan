# -*- coding: utf-8 -*-
"""pRRTC — GPU-parallel RRT-Connect (sampling paradigm).

Massively parallel RRT-Connect on the shared GPU collision substrate. Joint-
space start→goal only. This is the slim, self-contained sibling of the
production plugin: no constraints, no mesh path, no external fallback.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..base.planner_base import PlannerType, PlanningRequest, PlanningResult
from ._substrate_planner_base import SubstratePlannerBase

logger = logging.getLogger(__name__)


class PRRTCPlanner(SubstratePlannerBase):
    """GPU-parallel RRT-Connect planner."""

    _DEFAULT_YAML = "config/planners/prrtc.yaml"

    def __init__(self, config=None, config_path=None):
        super().__init__(PlannerType.PRRTC, config=config, config_path=config_path)

    def _init_buffers(self) -> None:
        import prrtc
        rm = self._robot_model
        self._solver_bufs = prrtc.SolverBuffers.create(
            max_samples=self._config.get("max_samples", 1000000),
            n_dof=rm.n_dof,
            num_new_configs=self._config.get("num_new_configs", 600),
            granularity=self._config.get("granularity", 16),
            n_joints=rm.n_joints,
            enable_mesh=False,
        )

    def _build_settings(self):
        import prrtc
        settings = prrtc.Settings()
        settings.num_new_configs = self._config.get("num_new_configs", 600)
        settings.granularity = self._config.get("granularity", 16)
        settings.range = self._config.get("range", 0.5)
        for key in ("max_samples", "max_iters", "balance", "tree_ratio",
                    "dynamic_domain", "dd_alpha", "dd_radius", "dd_min_radius",
                    "shortcut_path", "valid_segment_length", "collision_margin",
                    "shortcut_collision_margin", "time_limit_ms"):
            if key in self._config:
                setattr(settings, key, self._config[key])
        return settings

    def plan(self, request: PlanningRequest) -> PlanningResult:
        err = self._guard_request(request)
        if err is not None:
            return err

        import prrtc
        start_time = time.time()
        start = np.asarray(request.start_joint_state, dtype=np.float32)
        goal = np.asarray(request.target_joint_state, dtype=np.float32)
        goals = goal.reshape(1, -1)
        settings = self._build_settings()

        try:
            with self._solve_context() as scene_coll:
                if scene_coll is None:
                    return PlanningResult(
                        success=False, status="PlanningError",
                        error_message="pRRTC requires scene collision data "
                                      "(the world has no obstacles)",
                        solve_time=time.time() - start_time)

                flags = self._preflight(scene_coll, np.vstack([start, goal]))
                if bool(flags[0]):
                    return PlanningResult(
                        success=False, status="StartInCollision",
                        error_message="pRRTC start configuration in collision",
                        solve_time=time.time() - start_time)
                if bool(flags[1]):
                    return PlanningResult(
                        success=False, status="GoalInCollision",
                        error_message="pRRTC goal configuration in collision",
                        solve_time=time.time() - start_time)

                result = prrtc.solve_scene(
                    start, goals, scene_coll, settings,
                    self._robot_model, bufs=self._solver_bufs)
        except Exception as e:  # noqa: BLE001
            logger.exception("[pRRTC] solver raised")
            return PlanningResult(
                success=False, status="PlanningError",
                error_message=f"pRRTC planning failed: {e}",
                solve_time=time.time() - start_time)

        solve_time = time.time() - start_time
        if not result.solved:
            logger.warning("[pRRTC] no solution: iters=%s wall_time=%.1fms",
                           result.iters, result.wall_time_ms)
            return PlanningResult(
                success=False, status="PlanningFailed",
                error_message="pRRTC could not find a solution",
                solve_time=solve_time,
                extra_info={"solver": "prrtc", "iters": result.iters,
                            "wall_time_ms": result.wall_time_ms})

        path = np.asarray(result.path, dtype=np.float32)
        logger.info("[pRRTC] solved: cost=%.4f iters=%s waypoints=%d",
                    result.cost, result.iters, len(path))
        return self._finalize(
            path, request, solve_time,
            extra_info={"solver": "prrtc", "iters": result.iters,
                        "cost": result.cost, "wall_time_ms": result.wall_time_ms,
                        "geometric_waypoints": len(path)})
